# record-similarity

A gRPC service that answers one question about an edit to a record: **is this still the same record?**

Marketplace listings get "recycled": a merchant takes an existing product, say a blueberry gummy, and edits it into
an entirely different one, keeping the id and everything attached to it (reviews, rankings, history). This service
embeds each revision of a record field by field with a small sentence-embedding model running on
[Burn](https://burn.dev), compares it with the stored version, and accepts or rejects the change. A record's
quantities (weight, volume, potency) are compared beside the text, as numbers: converted to a common unit and
reported as "how many times larger or smaller". Accepted vectors are kept in [Qdrant](https://qdrant.tech), which
also makes the catalog searchable by similarity.

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
quantity  thc_per_item        x1.00  Unchanged (mg to g)
quantity  weight_per_item     x1.00  Unchanged (g to g)

$ just client evaluate sku-1 data/examples/vape.json                # a different product under the same id
verdict   Rejected (stored: false, revision 2)
overall   0.562 against a threshold of 0.900
drift     0.567 similarity to the original record
field     brand           0.642  Modified
field     category        0.559  Modified
field     description     0.635  Modified
field     name            0.509  Modified  << below floor
quantity  thc_per_item            -  Incomparable (g to %)
quantity  weight_per_item    x40.00  Modified (g to g)
measures  changed
reason    below_threshold              0.562 < 0.900
reason    field_below_floor name       0.509 < 0.800
reason    anchor_drift                 0.567 < 0.850
```

The edited gummies give their THC as `0.1 g` where the original said `100 mg`: the same amount, so unchanged. The
vape's weight is a fortieth of the gummies' and its THC is a percentage, which cannot be compared with a mass at
all. Those three lines are reported, not enforced: the example configuration sets no limit on any quantity (see
[Choosing thresholds](#choosing-thresholds)), and the three reasons are all about the text. A `reason` line is the
reason's code, what it is about if it is about one field, and the number against the limit it was held to.

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
Four things stand between this and enforcing on production traffic:

1. **Thresholds are calibrated on synthetic data.** `data/calibration.sample.jsonl` holds 46 pairs written for this
   repository. Collect labeled edits from production (`docs/data-request-prompt.md` describes exactly what to
   collect), run `calibrate` on them, and set the thresholds from that. Start in `DRY_RUN`.
2. **The choice of model is a prior, not a measured win.** On the synthetic pairs the default model and
   `all-MiniLM-L6-v2` are close. Real pairs should decide ([Using a different model](#using-a-different-model)).
3. **No quantity limit has been measured.** Quantities are compared and reported in every response, and reject
   nothing until a limit is configured. Where a limit belongs has to be read from the same labeled edits.
4. **Per-record locking is per process.** See [Limitations](#limitations).

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
| `jq` | Reading NDJSON logs and results at a terminal. | `brew install jq` |

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
just test-real            # parity with PyTorch, the pinned verdicts, then the whole stack through gRPC
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
| `record-similarity import [--config config.toml] EXPORT.jsonl [--check]` | Loads a whole catalog into the store. See [Importing a catalog](#importing-a-catalog). |
| `record-similarity compare [--config config.toml] BEFORE.json AFTER.json` | Scores one before/after pair offline. |
| `record-similarity calibrate [--config config.toml] PAIRS.jsonl [--full] [--split S] [--model DIR] [--ndjson]` | Scores labeled pairs and suggests a threshold. See [Choosing thresholds](#choosing-thresholds). |
| `record-similarity calibrate-diff OLD.ndjson (NEW.ndjson \| --config-b CONFIG) [--fail-on WHAT]` | Compares two calibrations, or one under a second configuration. See [Comparing two calibrations](#comparing-two-calibrations). |
| `record-similarity runs list [--runs DIR] [--kind KIND] [--limit N]` | Lists what was recorded: every calibration and every training run. See [Recorded runs](#recorded-runs). |
| `record-similarity pairs carve PAIRS.jsonl --fraction F [--from S] [--into S] [--salt TEXT]` | Gives a share of one split's owners a split of their own. See [Choose a model, then measure it once](#3-choose-a-model-then-measure-it-once). |

Environment variables:

| Variable | Effect |
|---|---|
| `QDRANT_API_KEY` | The API key for a secured Qdrant or Qdrant Cloud. Never read from the config file. |
| `RUST_LOG` | Log filter, `info` by default. `RUST_LOG=warn` quiets the per-request lines; `RUST_LOG=debug` adds detail. |
| `RECORD_SIMILARITY_RUNS` | Where runs are recorded, `runs` by default. The same as `--runs` on `calibrate`, `runs list`, and the trainer. |

**Logs are NDJSON.** Both binaries log one JSON object per line on stderr and nothing else: there is no text
format to switch to. Each line has `timestamp`, `level`, `message`, and `target`, with the event's own fields
beside them:

```json
{"timestamp":"2026-11-11T05:52:42.870451Z","level":"INFO","message":"serving gRPC","address":"127.0.0.1:50051","target":"record_similarity"}
{"timestamp":"2026-11-11T05:52:43.254934Z","level":"INFO","message":"evaluate","id":"sku-1","verdict":"Created","stored":true,"revision":1,"measure_changed":false,"target":"similarity_server::grpc"}
```

A command that fails ends with a `failed` line at level `ERROR` carrying the reason in `error`, and a crash is
reported the same way as `panicked`. Neither can be filtered out by `RUST_LOG`. To read logs at a terminal, pipe
them through `jq`: `just serve 2>&1 | jq -r '[.level, .message, (del(.timestamp, .level, .message, .target) |
tostring)] | join("  ")'`. The reports that `model fetch`, `compare`, and `calibrate` print on stdout are for a
person and stay as text. Usage errors for a mistyped command line are text too.

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

A record is an optional `id`, an optional `tenant`, a map of field name to text, and a map of quantity name to
quantity. The names must be ones the server's schema defines; `Describe` lists both kinds.

**Quantities.** A quantity is `{min, max, unit}`. A single value is sent with `min` equal to `max`, and so is a
range of which only one end is known. `unit` is free text: mass (`mcg`, `mg`, `g`, `kg`, `oz`, `lb`), volume (`ml`,
`cl`, `l`, `fl oz`), and `%` are converted before comparing, whatever their case or spacing, so `1 g` and
`1000 mg` are the same quantity. Any other unit is accepted and is comparable only with itself. Values must be
finite, not negative, and `min` not above `max`. A quantity is never sent as text in `fields`, and a quantity the
record does not have is left out. [Mapping your records](#mapping-your-records) shows where each column goes.

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

`fields.json` is one JSON object holding the record's text fields and its quantities together, like the files in
`data/examples/`. A string value is a text field. An object value is a quantity, written either as
`{"value": 3.5, "unit": "g"}` or as `{"min": 18, "max": 22, "unit": "%"}`:

```json
{
  "name": "Sour Diesel Vape Cartridge 1g",
  "category": "Vape Pens > Cartridges",
  "weight_per_item": {"value": 1, "unit": "g"},
  "thc_per_item": {"min": 85, "max": 90, "unit": "%"}
}
```

`compare` and `calibrate` read records in the same shape.

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
    },
    "quantities": {
      "weight_per_item": {"min": 40, "max": 40, "unit": "g"},
      "thc_per_item": {"min": 100, "max": 100, "unit": "mg"}
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
    },
    "quantities": {
      "weight_per_item": {"min": 1, "max": 1, "unit": "g"},
      "thc_per_item": {"min": 85, "max": 90, "unit": "%"}
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
  "revision": "1",
  "quantities": [
    { "field": "thc_per_item", "change": "QUANTITY_CHANGE_INCOMPARABLE", "incomparable": true, "previousUnit": "mg", "unit": "%" },
    { "field": "weight_per_item", "change": "QUANTITY_CHANGE_MODIFIED", "factor": 40, "previousUnit": "g", "unit": "g" }
  ],
  "measureChanged": true,
  "reasonDetails": [
    { "code": "REASON_CODE_BELOW_THRESHOLD", "value": 0.5413923263549805, "limit": 0.8999999761581421 },
    { "code": "REASON_CODE_FIELD_BELOW_FLOOR", "subject": "name", "value": 0.5116651654243469, "limit": 0.800000011920929 },
    { "code": "REASON_CODE_ANCHOR_DRIFT", "value": 0.5413923263549805, "limit": 0.8500000238418579 }
  ]
}
```

The quantities are reported and, with no limit configured, are not among the reasons. `reasonDetails` is the
same three reasons as data, in the same order as the sentences. Its numbers are the response's 32-bit values
written out as 64-bit ones, which is why the threshold `0.9` reads `0.8999999761581421` there: narrowed back to
32 bits they are equal to `similarity` and `threshold` exactly.

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
    quantities: [(
      "weight_per_item".to_owned(),
      pb::Quantity { min: 40.0, max: 40.0, unit: "g".to_owned() },
    )]
    .into(),
  }),
  mode: pb::WriteMode::DryRun.into(),
  ..Default::default()
};
let response = client.evaluate(request).await?.into_inner();
if response.verdict() == pb::Verdict::Rejected {
  for (detail, sentence) in response.reason_details.iter().zip(&response.reasons) {
    match detail.code() {
      pb::ReasonCode::FieldBelowFloor => eprintln!("the {:?} changed too much", detail.subject),
      _ => eprintln!("rejected: {sentence}"),
    }
  }
}
for quantity in &response.quantities {
  eprintln!("{}: {:?}, factor {:?}", quantity.field, quantity.change(), quantity.factor);
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
record = pb.Record(
    id="sku-1",
    tenant="merchant-1",
    fields={"name": "Blueberry Gummies 100mg"},
    quantities={"weight_per_item": pb.Quantity(min=80, max=80, unit="g")},
)

response = client.Evaluate(pb.EvaluateRequest(record=record, mode=pb.WRITE_MODE_DRY_RUN))
print(pb.Verdict.Name(response.verdict), response.similarity, list(response.reasons))
for reason in response.reason_details:
    print(pb.ReasonCode.Name(reason.code), reason.subject, reason.value, reason.limit)
for quantity in response.quantities:
    print(quantity.field, pb.QuantityChange.Name(quantity.change), quantity.factor)
```

### What `Evaluate` returns

| Request | Verdict | Stored? |
|---|---|---|
| No `id` | `VERDICT_EMBEDDED`: vector computed, nothing compared | never |
| `id` with no baseline | `VERDICT_CREATED`: this is the baseline now | yes |
| `id`, similar enough | `VERDICT_ACCEPTED` | yes (skipped if nothing changed) |
| `id`, too different | `VERDICT_REJECTED`, with `reasons` and `reason_details` | no: the baseline is untouched |

`mode` changes the write, never the verdict:

| Mode | Behavior |
|---|---|
| `WRITE_MODE_ENFORCE` (default) | As in the table above. |
| `WRITE_MODE_DRY_RUN` | Never writes. |
| `WRITE_MODE_FORCE` | Always writes, and makes this revision the new reference point for drift. The verdict is still reported. |

Response fields: `similarity` (overall, absent when there was no baseline), `threshold` (the one applied),
`fields` (per field: `change`, `similarity`, `weight`, `below_floor`), `anchor_similarity` (similarity to the
first value each of the record's fields had), `stored`, `reasons`, `reason_details`, `revision`, `quantities` (per
quantity, below), `measure_changed`, and `vector` if requested. `threshold` on the request overrides the
configured overall threshold for that one call.

`reasons` are sentences for a person; `reason_details` are the same reasons as data, one per sentence and in the
same order. Each has a `code` (`REASON_CODE_BELOW_THRESHOLD`, `_FIELD_BELOW_FLOOR`, `_ANCHOR_DRIFT`,
`_QUANTITY_BEYOND_FACTOR`, `_QUANTITY_INCOMPARABLE`, `_MEASURE_CHANGED`), and where they apply a `subject`, a
`value`, and a `limit`. The `subject` is the name of an entry in `fields` or `quantities` of the same response,
where the change and the units are; `value` is the similarity or the factor that was measured. The wording of a
sentence may change and a code will not, so branch on the code. A client that meets a code it does not know (a
newer server) should show the sentence at the same position.

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

Each entry of `quantities` describes one quantity that either version has:

| Field | Meaning |
|---|---|
| `change` | One of the values below. |
| `factor` | How many times larger or smaller the amount became, after unit conversion; for a range, whichever end moved more. Always at least 1, so "eight times bigger" and "an eighth" are both 8, and exactly 1 when nothing changed. Absent when there is no number to give. |
| `zero_crossing` | The amount moved between zero and a value above zero, so no factor describes it. |
| `incomparable` | The two values are in units that cannot be converted into one another. |
| `previous_unit`, `unit` | The unit of the value compared against, and the record's. |
| `beyond_limit` | This quantity exceeds a limit the schema sets for it, which rejects the record. Never set when no limit is configured. |

| Change | Meaning |
|---|---|
| `QUANTITY_CHANGE_UNCHANGED` | The same amount, in whatever unit. `1 g` and `1000 mg`. |
| `QUANTITY_CHANGE_MODIFIED` | Both versions have a value, in comparable units, and the amounts differ. |
| `QUANTITY_CHANGE_INCOMPARABLE` | Both versions have a value, in units that cannot be compared (`%` and `mg`). |
| `QUANTITY_CHANGE_ADDED` | The record has never had this quantity. Taken on trust, as a new text field is. |
| `QUANTITY_CHANGE_REMOVED` | The stored version has it and this one does not. |
| `QUANTITY_CHANGE_RESTORED` | The stored version has none, but the record once did, and has one again. Compared with the first value it had. |

`measure_changed` is set when the record changed which measures it is described by: a quantity became
incomparable, or one quantity was removed as a different one appeared (a weight for a volume). Filling in a
quantity that was merely missing does not set it.

A quantity never moves `similarity`. Whether it can reject is a matter of configuration, and by default it cannot
([How a verdict is reached](#how-a-verdict-is-reached)). A change in a quantity alone is still a change to the
record: an accepted one is stored and counts as a revision.

### Errors

| Status | Meaning | What to do |
|---|---|---|
| `INVALID_ARGUMENT` | The record does not fit the schema (unknown field or quantity, missing required field, blank id, a quantity that is negative, not finite, has `min` above `max`, or has no unit), a parameter is out of range, or `mode` is a value this server does not know. | Fix the request. Retrying will not help. |
| `NOT_FOUND` | `FindSimilar` by an id that is not stored. | |
| `FAILED_PRECONDITION` | The stored record belongs to a tenant the request does not name (it names another, or none), or was written under a different model or field set. | Send the record's tenant, or see [Re-indexing](#re-indexing). |
| `RESOURCE_EXHAUSTED` | The embedding queue is full. | Retry with backoff. |
| `UNAVAILABLE` | Qdrant cannot be reached. | Retry with backoff. |

Nothing is written when an error is returned.

## Integrating it

**Bootstrapping an existing catalog.** Export every record as one JSON object per line and load the file with
`record-similarity import` ([Importing a catalog](#importing-a-catalog)). Every record that is not stored yet
becomes a baseline. On a laptop CPU, 2,000 four-field records were imported in 19 seconds, so 50,000 take about
eight minutes. Calling `Evaluate` once per record does the same thing one request at a time, for a caller that
cannot run a command next to Qdrant: each call returns `VERDICT_CREATED`. Keep fewer requests in flight than
`model.queue_depth` (256 by default) and back off on `RESOURCE_EXHAUSTED`.

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

### Mapping your records

The service has two kinds of field: text, which is embedded, and quantities, which are compared as numbers. For a
brand's product record, with the example configuration:

| On the record | In the request | Notes |
|---|---|---|
| Brand foreign key | `tenant`, as a string | The tenant is the brand that owns the product. A product moving to another brand is then refused by the tenant check. |
| Brand name, looked up from the key | `fields["brand"]` | The same for every product of a brand, so it rarely moves a score. |
| Root and sub category, through the join | `fields["category"]`, for example `Edibles > Gummies` | See the two rules below. |
| `name`, `description` | `fields["name"]`, `fields["description"]` | Sent as stored. Do not trim or clean them; the service normalizes whitespace itself. |
| `weight_per_item_value` and `_unit` | `quantities["weight_per_item"]` | A single value: `min` and `max` both set to it. |
| `volume_per_item_value` and `_unit` | `quantities["volume_per_item"]` | A single value. |
| `thc_per_item_min_value`, `_max_value`, `_unit` | `quantities["thc_per_item"]` | A range. |
| `cbd_per_item_min_value`, `_max_value`, `_unit` | `quantities["cbd_per_item"]` | A range. |

Twelve columns become two text fields and four quantities. The quantity names are configuration, not code: a
fifth (a pack size, say) is a new `[quantities.<name>]` table and nothing else.

Rules for the caller:

- **Category text is built from names, never ids**, root first and then the sub category. An id says nothing to a
  text model, and two ids are as different as any other two.
- **Category text is built in a fixed order.** If a product can have several category rows, sort them the same way
  every time. Otherwise a product whose rows merely came back in another order looks edited.
- **A quantity is never rendered into text.** Not as a field of its own and not appended to the name. What the
  name already says about size or potency stays in the name; the columns go in `quantities`.
- **A null column is a quantity left out**, not a zero. A stored zero is a value and is sent as one. A move
  between zero and anything else is reported as a `zero_crossing`.
- **Send the unit as stored.** The service converts the units it knows. A unit it does not know is still
  accepted, and compared with itself.

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

**Quantities are judged beside this, never inside it.** They are not embedded and do not enter the weighted
mean, so every threshold above means the same with or without them (D16 has the measurements behind that). For
each quantity the schema names:

1. Both values are converted to their dimension's base unit and compared. The result is a **factor**, the larger
   over the smaller: 1 for the same amount, 8 for `3.5 g` against `28 g` in either direction. A unit of another
   dimension makes the pair **incomparable** instead.
2. A quantity that was removed earlier and has a value again is compared with the first value it had, for the
   reason a text field is: otherwise it could be swapped in two steps.
3. The revision is additionally rejected if, and only if, the configuration says so:
   - the factor is above that quantity's `reject_above_factor` (a move between zero and a value above zero
     counts as above any limit);
   - the quantity became incomparable and has `reject_incomparable`;
   - the record's measures changed and `policy.reject_on_measure_change` is set.

With none of those configured, which is how the example configuration ships, quantities are reported in every
response and change no verdict.

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
just calibrate path/to/pairs.full.jsonl config.toml --full --split holdout     # a data export; flags go last
```

The input is one JSON object per line, labeled `edit` (should be accepted) or `recycle` (should be rejected).
Blank lines and lines starting with `#` are ignored. `before` and `after` are records in the shape the bundled
client reads: a string is a text field, an object is a quantity. There are two formats:

- **Strict**, the default, for pairs written by hand. Four keys per line (and optionally `anchor`, below), and
  every name in a record must be one the schema defines, so a typo is an error.

  ```json
  {"label": "recycle", "note": "an eighth becomes an ounce", "before": {"name": "...", "weight_per_item": {"value": 3.5, "unit": "g"}}, "after": {"name": "...", "weight_per_item": {"value": 28, "unit": "g"}}}
  ```

- **Full** (`--full`), for a data export: the `pairs.full.jsonl` that
  [`docs/data-request-prompt.md`](docs/data-request-prompt.md) asks for. Extra keys are allowed. Each record is
  reduced to what the schema defines, and what that leaves out is counted by key in the report, so a misspelled
  key shows up as a number and not as a quietly weaker score. `pair_id`, `owner`, `split`, `variant_change`,
  `label_source`, and `label_confidence` are used when present. A pair with any label other than `edit` or
  `recycle` (an `unsure`) is scored and listed, and counted nowhere.

| Flag | Meaning |
|---|---|
| `--full` | Read the full format. |
| `--split NAME` | With `--full`: score only the pairs of this split. |
| `--min-confidence high\|medium\|low` | With `--full`: leave out labeled pairs whose `label_confidence` is lower. Default `medium`. |
| `--ndjson` | Write the calibration as NDJSON events in place of the report ([below](#the-calibration-as-data)). |
| `--seed N` | Fixes the resampling behind the intervals. Default 0. |

The report starts with the model and the width it ran at, lists every pair with its score and verdict, flags the
ones the current config gets wrong, and ends with a summary. On the bundled sample:

<!-- measured: calibrate data/calibration.sample.jsonl -->
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

quantity (factor)    in edits                               in recycles
cbd_per_item         1 to or from zero                      -
thc_per_item         1.01 to 1.14 ( 4)                      1.04 to 10.00 ( 6), 2 incomparable
volume_per_item      1.01 to 1.01 ( 1)                      -
weight_per_item      1.01 to 1.01 ( 1)                      2.00 to 40.00 ( 5)
measures changed in 0 of 26 edit(s) and 4 of 20 recycle(s)
cbd_per_item: 1 edit(s) move it to or from zero, which any factor limit rejects
a limit of 1.16 on thc_per_item is above every edit (largest 1.14) and rejects 5 of the 6 recycle(s) that change it
a limit of 1.42 on weight_per_item is above every edit (largest 1.01) and rejects 5 of the 5 recycle(s) that change it

by category               edits  recycles  wrongly rejected  wrongly accepted
Concentrates                  2         2                 0                 0
Edibles                      15         8                 0                 0
Flower                        4         4                 0                 0
Pre-Rolls                     2         2                 0                 0
Tinctures                     1         2                 0                 0
Vape Pens                     2         2                 0                 0

configured policy (threshold 0.900 plus field floors and any quantity limits): 0 edit(s) wrongly rejected, 0 recycle(s) wrongly accepted, of 46 pairs
best overall threshold for this data: 0.895 (margin 0.046; 0 edit(s) rejected, 0 recycle(s) accepted)
an edit scores above a recycle in 100.0% of pairings (AUC 1.0000): the number to compare two models by, since it needs no threshold
to accept at most 1.0% of recycles on the overall score alone: threshold 0.895, rejecting 0.0% of edits
to accept at most 5.0% of recycles on the overall score alone: threshold 0.846, rejecting 0.0% of edits
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
- **`reject_above_factor` for a quantity** belongs above the largest factor that quantity shows in any edit. A
  quantity limit is an extra way to reject, beside the text checks, so what matters is that it rejects no edit;
  most recycles leave a quantity alone (another strain at the same weight) and are for the text to catch. Each
  cell is the range of factors over the pairs that changed the amount, with their count; the same amount in
  another unit is no change and is not counted. The `a limit of ...` line does the arithmetic: the midpoint
  between the largest factor in edits and the next factor above it in recycles, and how many recycles that would
  reject. In the sample, one recycle changes the potency by 1.04, less than an ordinary batch update, and no
  limit on `thc_per_item` would catch it.
- **`reject_incomparable`** is safe for a quantity when `incomparable` never appears for it under `in edits`.
- **`policy.reject_on_measure_change`** is safe when the `measures changed` line counts no edits.
- **`to or from zero`** deserves a look before any factor limit is set, because such a change is beyond every
  limit. In the sample it is an edit: a CBD content that went from 0 to 0.1%.
- **The slices** show where the errors are. `by category` is always there. With `--full` the pairs are also
  sliced by `variant_change`, `label_source`, and `owner` (the ten largest, to show one brand dominating).
- **`to accept at most N% of recycles`** is the trade the overall threshold makes: the lowest threshold that lets
  through no more than that share of recycles, and the share of edits it rejects. It looks at the overall score
  only, without floors or quantity limits. A line is left out when it allows no more recycles than the one before
  it, which with fewer than a hundred recycles is most of them.

**The quantity lines of the sample are there to show the report, not to be read from.** Nineteen of the 46
synthetic pairs carry quantities, written by hand. That is why `config.example.toml` sets no quantity limit at
all, and why none should be set until this section has been produced from labeled production edits.

### A pair that is one step of a history

A before/after pair cannot show the two-step swap that the anchor exists to stop: blank a field in one edit, fill
it with another product's text in the next. Scored as a pair, the second step is only a field being filled in,
which is left out of the score. So a pair, in either format, may carry a third record, `anchor`: the record as it
was first created.

```json
{"label": "recycle", "note": "brand blanked earlier, now another brand", "anchor": {"name": "Blueberry Gummies", "brand": "Sunny Orchard"}, "before": {"name": "Blueberry Gummies"}, "after": {"name": "Blueberry Gummies", "brand": "Volt Labs"}}
```

It is then judged as the server judges a record imported with that original version and revised once: a field or
quantity `before` lacks is compared with the anchor (`restored`), and the similarity to the anchor is reported
and held to `anchor_threshold`. The report marks such a pair `(anchored)`. A pair without an `anchor` is scored
exactly as before.

### A data export, with intervals

`data/calibration.sample.full.jsonl` is the same sample in the full format, with invented owners, splits, and
label sources, three `unsure` variant changes, and a `strain` key the schema does not define. Its report has more
in it (shortened here):

<!-- measured: calibrate data/calibration.sample.full.jsonl --full -->
```text
$ just calibrate data/calibration.sample.full.jsonl config.example.toml --full
...
unsure     0.973  accepted  gummies doubled in potency, same flavor
unsure     0.938  accepted  an eighth becomes a quarter
unsure     0.911  accepted  a single pre-roll becomes a two pack
...
others    overall similarity from 0.911 to 0.973 (3 pair(s) with another label, 3 of which would be accepted; counted nowhere below)
...
by variant change         edits  recycles  wrongly rejected  wrongly accepted
count                         0         1                 0                 0
flavor                        0         1                 0                 0
strain                        0         4                 0                 0
...
by owner (largest)        edits  recycles  wrongly rejected  wrongly accepted
brand-1                      13         6                 0                 0
brand-2                       4         4                 0                 0
...
configured policy (threshold 0.900 plus field floors and any quantity limits): 0 edit(s) wrongly rejected, 0 recycle(s) wrongly accepted, of 45 pairs
false-reject rate 0.0% (0.0% to 0.0%), false-accept rate 0.0% (0.0% to 0.0%) (95% intervals from resampling 7 owners)
best overall threshold for this data: 0.895 (margin 0.046; 0 edit(s) rejected, 0 recycle(s) accepted)
an edit scores above a recycle in 100.0% of pairings (AUC 1.0000): the number to compare two models by, since it needs no threshold
to accept at most 1.0% of recycles on the overall score alone: threshold 0.895, rejecting 0.0% (0.0% to 0.0%) of edits
to accept at most 5.0% of recycles on the overall score alone: threshold 0.846, rejecting 0.0% (0.0% to 0.0%) of edits

record keys that were not compared: strain (29, not in the schema)
left out: 0 pair(s) of another split, 1 labeled pair(s) below the confidence asked for
```

- **The intervals** are the middle 95% of each rate when the **owners** are resampled, 1,000 times, with all
  their pairs. Pairs from one brand are not independent of each other, so resampling pairs would make the
  intervals too narrow, and narrower the more one brand contributed. They need every labeled pair to name an
  owner and at least two owners, which is why the strict format has none. Zero errors give a zero-width
  interval, which says nothing about the pairs that were not sampled: with 45 pairs, treat it as no evidence.
- **The `unsure` pairs** are the ones to look at by hand. Here all three same-brand variant changes score above
  the threshold and would be accepted, which is the case the data request calls the hardest.
- **`record keys that were not compared`** should hold only keys you expect. A field you meant to compare turning
  up here is a misspelling in the export or a field missing from the config.

### The calibration as data

`--ndjson` writes the same calibration as events, one JSON object per line on stdout, for anything that is not a
person: a later run to compare against, a chart, a table. They go through the same logging as every other result,
so each line has `timestamp`, `level`, `message`, and `target` as well as the fields below, and `RUST_LOG` cannot
filter them out.

```sh
just calibrate data/calibration.sample.full.jsonl config.example.toml --full --ndjson > calibration.ndjson
jq -r 'select(.message == "pair" and .label == "unsure") | [.pair_id, .variant_change, .similarity, .accepted] | @tsv' calibration.ndjson
```

```text
p047	potency	0.973413	true
p048	size	0.938342	true
p049	count	0.910539	true
```

The events are flat: a pair is one `pair` line and then one line per field, quantity, and reason, all carrying
its `pair_id`. One recycle from the sample, with the envelope fields removed:

```json
{"message":"pair","pair_id":"p030","label":"recycle","split":"calibrate","owner":"brand-1","variant_change":"flavor","label_source":"moderator_revert","label_confidence":"high","category":"Edibles","note":"gummies become a different candy, same category and brand","similarity":0.808716,"accepted":false,"wrong":false,"measure_changed":false}
{"message":"pair_field","pair_id":"p030","field":"name","change":"modified","similarity":0.65489,"weight":4.0,"below_floor":true}
{"message":"pair_quantity","pair_id":"p030","quantity":"thc_per_item","change":"modified","factor":2.0,"zero_crossing":false,"incomparable":false,"previous_unit":"mg","unit":"mg","beyond_limit":false}
{"message":"pair_reason","pair_id":"p030","code":"field_below_floor","field":"name","value":0.65489,"limit":0.8}
```

| `message` | One per | Fields |
|---|---|---|
| `calibration` | run, first | `fingerprint`, `dimension`, `max_tokens`, `threshold`, `anchor_threshold`, `reject_on_measure_change`, `pairs_sha256`, `pairs`, `split`, `min_confidence`, `seed` |
| `pair` | pair | `pair_id`, `label`, `split`, `owner`, `variant_change`, `label_source`, `label_confidence`, `category`, `note`, `similarity`, `accepted`, `wrong`, `measure_changed`, `anchored`, `anchor_similarity` |
| `pair_field` | pair and field | `pair_id`, `field`, `change`, `similarity`, `weight`, `below_floor` |
| `pair_quantity` | pair and quantity | `pair_id`, `quantity`, `change`, `factor`, `zero_crossing`, `incomparable`, `previous_unit`, `unit`, `beyond_limit` |
| `pair_reason` | pair and reason for rejecting it | `pair_id`, `code`, `field`, `value`, `limit` |
| `field_range` | field and label | `field`, `label`, `count`, `low`, `high` |
| `quantity_range` | quantity and label | `quantity`, `label`, `count`, `low`, `high`, `incomparable`, `zero_crossings` |
| `suggestion` | run | `threshold`, `margin`, `false_rejects`, `false_accepts` |
| `quantity_suggestion` | quantity | `quantity`, `reject_above_factor`, `edits_high`, `recycles_next`, `recycles_caught`, `recycles_changed`, `edit_zero_crossings` |
| `slice` | attribute and value | `by`, `value`, `edits`, `recycles`, `false_rejects`, `false_accepts`, `edits_low`, `edits_high`, `recycles_low`, `recycles_high` |
| `operating_point` | target | `false_accept_target`, `threshold`, `false_accept_rate`, `false_reject_rate`, `false_reject_low`, `false_reject_high` |
| `ignored_key` | key and reason | `key`, `reason`, `count` |
| `summary` | run, last | `pairs`, `edits`, `recycles`, `unlabeled`, both error counts and rates with `_low` and `_high`, `owners`, `skipped_split`, `skipped_confidence` |

A value that does not apply is left out, never `null`. The reason codes are `below_threshold`,
`field_below_floor`, `anchor_drift`, `quantity_beyond_factor`, `quantity_incomparable`, and `measure_changed`,
and the API's `reason_details` carry the same six as `REASON_CODE_` plus the code in capitals.
A pair's `pair_id` is the one the file gave it; a pair without one gets a fingerprint of its content, so the same
pair has the same id in every run. Similarities are rounded to six places. `schema_field` and `schema_quantity`
lines after the first say which fields and quantities the run was scored with, and their weights and limits.

### Comparing two calibrations

Keep a run as a file, and the next one can be held against it: did the new model, thresholds, weights, or code
change any verdict?

```sh
just calibrate pairs.full.jsonl config.toml --full --ndjson > old.ndjson
just calibrate pairs.full.jsonl config-new.toml --full --ndjson > new.ndjson
just calibrate-diff old.ndjson new.ndjson
```

For a change of **weights, floors, thresholds, or quantity limits** the second run is not needed. Those are all
applied to measurements the first run already wrote down, so `--config-b` judges its pairs again under another
configuration without loading a model or embedding anything. It takes a fraction of a second, which makes trying
a setting cheap. Here the sample is judged again with the threshold raised to 0.95 and the weight of `name`
doubled:

```text
$ just calibrate-diff old.ndjson --config-b config-b.toml
old    f46e48ee0ea2dd02-d256-t256, threshold 0.900, 48 pairs
new    the old run judged again under config-b.toml: threshold 0.950
pairs  48 compared (0 only in the old run, 0 only in the new)

label      old                new           what              pair
unsure   0.938 accepted  -> 0.912 rejected  changed           p048 an eighth becomes a quarter
edit     0.950 accepted  -> 0.928 rejected  NEW FALSE REJECT  p014 flower weight written out
unsure   0.911 accepted  -> 0.910 rejected  changed           p049 a single pre-roll becomes a two pack

                                 old     new
edits wrongly rejected             0       1
recycles wrongly accepted          0       0
best overall threshold         0.895   0.857
its margin                     0.046   0.072

slices whose errors changed
category Flower: wrongly rejected 0 -> 1, wrongly accepted 0 -> 0
label source identifiers_unchanged: wrongly rejected 0 -> 1, wrongly accepted 0 -> 0
owner brand-2: wrongly rejected 0 -> 1, wrongly accepted 0 -> 0
```

Reading it: every pair whose verdict changed is listed, the largest change in score first, and named for what it
means given its label. Raising the threshold costs one edit; the heavier `name` also moves two of the `unsure`
variant changes below it, and widens the margin a best threshold would have. A real second run under the same
second configuration prints the same three lines, which is the check that judging again and scoring again agree.

- `--config-b` reads only `[policy]`, `[fields]`, and `[quantities]`. The field and quantity names must be the
  ones the run was scored with. A change of **model or of the set of fields** changes the measurements themselves
  and needs a real second run.
- Two runs are compared only if they were made from the same selection of the same pairs file (the same file,
  `--split`, and `--min-confidence`). `--intersect` compares the pairs they share and counts the rest.
- `--fail-on new-false-accepts`, `any-flip`, or `errors-up` makes the command fail (exit status 1, with a `failed`
  line saying why) when the comparison shows that, for use in a script. `--ndjson` writes one `flip` line per
  changed verdict and a closing `diff` line in place of the report.
- Pairs with any other label are listed when their verdict changes and never count as an error either way.
- A verdict can stay the same while what is behind it moves. The report ends with every pair whose overall, anchor,
  or field similarity moved by more than `--tolerance` (0.001 unless given), whose field or quantity changed in
  kind (`added` to `restored`), whose factor changed, or which is now rejected for another reason.
  `--fail-on moved` fails on any of those, and `--fail-on` may be given more than once. With `--ndjson` each is a
  `moved` line (`pair_id`, `what`, `subject`, `old`, `new`, `old_change`, `new_change`) between the `flip` lines
  and the `diff` line, which counts the pairs in `moved`.

To hold a calibration of real pairs to an earlier one, keep the `--ndjson` output and gate on it:

```sh
just verdict-gate kept.ndjson pairs.full.jsonl config.toml    # fails on a flipped verdict or a score that moved
```

If the two overall ranges overlap on real data, there are three things to try, in order of cost: weights and
floors, [a different model](#using-a-different-model), and [fine-tuning](#fine-tuning-the-model).

`record-similarity compare before.json after.json` scores a single pair the same way, to see why one verdict came
out as it did. For the example gummies against the example vape:

```text
$ record-similarity compare --config config.example.toml data/examples/gummies.json data/examples/vape.json
model     f46e48ee0ea2dd02-d256-t256 (256 dimensions per field, up to 256 tokens)

field            weight similarity  change
brand              1.00      0.642  modified
category           3.00      0.559  modified
description        2.00      0.653  modified
name               4.00      0.512  modified  << below floor

quantity               factor  change         units
thc_per_item                -  incomparable   mg to %
weight_per_item         40.00  modified       g to g
the measures the record is described by changed

overall similarity 0.567 (threshold 0.900)
verdict: REJECTED
  - overall similarity 0.567 is below the threshold 0.900
  - field `name` similarity 0.512 is below its floor 0.800
```

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
- The workflow below has been run end to end on synthetic data, on the CPU and on Metal. That includes killing a
  run part-way and resuming it from its last checkpoint.
- It has not been used to produce a better model, because that needs real data. The CUDA build compiles but has
  never been run.

### 1. Assemble training data

Two kinds of signal, either or both. The formats are the ones `docs/data-request-prompt.md` asks for.

- **Labeled pairs** (`--pairs`): the same before/after pairs used for calibration. Extra keys are ignored, so
  `pairs.full.jsonl` works directly. Pairs labeled anything other than `edit` or `recycle` are skipped. If pairs
  carry a `split` key, `calibrate` is trained on (`--split` changes that), `holdout` is only ever measured
  (`--holdout-split`), and `validation` is what checkpoints are compared on (`--validation-split`; see
  [step 3](#3-choose-a-model-then-measure-it-once)). Pairs with no `split` are all trained on.
- **Catalog groups** (`--catalog-groups`): listings from different merchants linked to the same catalog product.
  These are naturally occurring paraphrases and usually the larger, cleaner source.

  ```json
  {"group": "g1", "catalog": {"name": "Blueberry Gummies 100mg"}, "listings": [{"name": "WYLD Blueberry 100mg (10pk)"}]}
  ```

By default only `name` and `description` are learned from (`--fields`). Categorical fields are left out because the
same change (say, sativa to hybrid) appears in both edits and recycles, which is noise at the level of one field.
Quantities are never learned from: they are not text, the server does not embed them, and the trainer skips any
value that is not a string.

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
| `--base` | (required) | The model directory to start from. Replaced by `--resume` when continuing a run. |
| `--out` | (required) | Where to write the fine-tuned directory. Put it under `models/`, which git ignores. It must not exist yet, or be empty: a model is never overwritten. |
| `--pairs`, `--catalog-groups` | | At least one. |
| `--fields` | `name,description` | Fields to learn from. |
| `--split` | `calibrate` | Which `split` of the pairs to train on. |
| `--holdout-split` | `holdout` | Which `split` to report on without training on it. |
| `--validation-split` | `validation` | Which `split` to compare checkpoints on. Never trained on. Used when the file has such pairs. |
| `--no-validation` | off | Do not compare checkpoints, even if the file has validation pairs. |
| `--select-width` | the narrowest of `--dims` | The embedding width checkpoints are compared at. Use the server's `output_dim`. |
| `--dims` | `1024,512,256` | Embedding widths to keep usable. Must include the server's `output_dim`. A width the model cannot produce is replaced by its native width, so a 384-wide model is trained at `384,256`. |
| `--epochs` | `2` | Passes over the data. |
| `--batch-size` | `32` | Larger batches give the catalog-group loss more negatives per step, and cost memory. |
| `--learning-rate` | `2e-5` | The peak, after a warm-up over the first 10% of steps. Lower it if the model gets worse on pairs held back from training. |
| `--max-tokens` | `256` | Match `model.max_tokens` in the server config. |
| `--device` | `cpu` | `cpu`, `metal`, or `cuda`. The GPU ones need a binary built with that feature. |
| `--seed` | `0` | Fixes the batch order and dropout. |
| `--no-dropout` | off | Trains without the encoder's dropout. |
| `--weight-decay` | `0` | Applied to every parameter. |
| `--max-grad-norm` | `1` | Global gradient-norm clipping; `0` disables it. |
| `--log-every` | `10` | Steps between progress lines. |
| `--checkpoint-dir` | `<out>.checkpoints` | Where checkpoints are written. Must not be inside `--out`. |
| `--checkpoint-every` | (none) | Also checkpoint every this many steps (at least 1). One is always written at the end of each pass. |
| `--keep-checkpoints` | `0` | Keep only this many of the run's checkpoints, deleting the oldest. `0` keeps all. |
| `--no-checkpoints` | off | Write none. |
| `--resume` | | A checkpoint directory to continue from, in place of `--base`. |

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

When training ends it saves the model, loads it back the way the server does, and reports where edits and
recycles sit before and after. This run trained on 47 pair and 15 catalog examples (synthetic) in 4 seconds on a
laptop CPU; these are two of the lines it wrote to stdout, shortened:

```json
{"message":"separation","split":"held_out","stage":"before","compared_with":"base model","width":256,"edits":5,"edits_mean":0.9629,"recycles":8,"recycles_mean":0.6444,"gap_of_means":0.3185}
{"message":"separation","split":"held_out","stage":"after","width":256,"edits":5,"edits_mean":0.9617,"recycles":8,"recycles_mean":0.6108,"gap_of_means":0.3509}
```

Those numbers are a progress check on single field texts, with no field weights, floors, or thresholds. They tell
you whether training moved things the right way; step 3 tells you whether the model is better.

**Output.** Everything the trainer writes is one JSON object per line (NDJSON), in two streams:

- **stdout: results.** One `separation` line per split (`training`, `held_out`), stage (`before`, `after`), and
  width, then one `finished` line. With validation pairs there is also one for the model the run started from,
  for every checkpoint, and for the final model, as each is scored. This is the run's output, so no log level
  hides it:
  `just finetune ... > results.ndjson`.
- **stderr: logs.** Progress, filtered by `RUST_LOG` (default `info`). A failure is always written here,
  whatever `RUST_LOG` says, as a `failed` line at level `ERROR` (or `panicked`, for a crash).

```json
{"timestamp":"2026-11-11T03:16:49.201435Z","level":"INFO","message":"training","step":10,"of":10,"epoch":2,"pair_loss":0.0171,"group_loss":0.4298,"grad_norm":12.734,"rate":0.00000222,"elapsed_s":4.0,"target":"record_similarity_train"}
{"timestamp":"2026-11-11T03:16:47.285904Z","level":"INFO","message":"checkpoint","step":5,"epoch":1,"path":"models/mdbr-leaf-mt-ft.checkpoints/step-000005","target":"record_similarity_train"}
```

| `message` | Stream | When | Fields |
|---|---|---|---|
| `labeled pairs`, `catalog groups` | stderr | Once, as the data is read. | Example counts, and how many pairs were skipped or held out. |
| `model loaded` | stderr | Once. | The model, device, widths, and checkpoint directory; `resumed_at` when resuming. |
| `training` | stderr | Every `--log-every` steps. | `step`, `of`, `epoch`, `pair_loss` and `group_loss` (means since the last line; left out if no batch of that kind fell in it), `grad_norm`, `rate`, `elapsed_s`. |
| `checkpoint` | stderr | Each checkpoint. | `step`, `epoch`, `path`. |
| `separation` | stdout | At the end. Only for a split that has pairs. | `split` (`training`, `held_out`, `validation`), `stage` (`before`, `after`; `checkpoint` on validation lines; `selected` on the held-out line of a chosen model that is not the final one), `step`, `width`; `edits` and `recycles` (counts) with `_min`, `_mean`, `_max` for each; `gap_of_means`; `auc` (how often an edit scores above a recycle); `path` where the line is about a model on disk. A `before` line also says what it describes: `compared_with` is `base model`, or `checkpoint` on a resumed run. |
| `finished` | stdout | At the end. | `out`, example counts, `steps`, `elapsed_s`, `resumed_at` if resumed. With validation pairs: `validation_examples`, `select_width`, and the chosen model as `best_step`, `best_model`, `best_auc`, `best_is_final`. |
| `failed` | stderr | In place of the results. | `error`. |

Losses, rates, and similarities are numbers, rounded, so they chart without parsing. For example, the loss curve:
`jq -c 'select(.message == "training") | [.step, .pair_loss, .group_loss]'`. A loss or a gradient that stops being
a finite number ends the run with a `failed` line before the step is applied, so none of these is ever `NaN`.
`just` and `cargo` print a line or two of their own on stderr; the binary itself writes only JSON, apart from
usage errors for a mistyped command line.

**Checkpoints.** A run saves itself at the end of every pass over the data (and every `--checkpoint-every`
steps, if set) into `<out>.checkpoints/step-000120` and so on. A checkpoint does two jobs:

- **It lets an interrupted run continue.** Give the same command again with `--resume` in place of `--base`:

  ```sh
  just finetune --resume models/mdbr-leaf-mt-ft.checkpoints/step-000120 --out models/mdbr-leaf-mt-ft \
    --pairs pairs.full.jsonl --catalog-groups catalog-groups.jsonl
  ```

  The run picks up at that step: the same batches in the same order, the same point in the learning-rate
  schedule, and the optimizer state it had. Every other option (`--max-tokens` included) and the data must be
  what the run was started with, and a checkpoint from a different run is refused rather than continued on a
  different schedule. With `--no-dropout` the resumed run ends on exactly the weights an uninterrupted one would
  have (checked on the real model: not one of the 101 tensors differs). With dropout it differs only in which
  units are dropped after the resume.
- **It is a complete model directory.** Point a config's `model.dir` at a checkpoint and calibrate it like any
  other model (step 3). That is how to pick the best pass of a long run instead of assuming it is the last.

A checkpoint of the default model is about 265 MB: the weights (90 MB) plus the optimizer's state (184 MB), which
is two running averages for every weight. Some things worth knowing:

- **Disk.** `--keep-checkpoints N` keeps the newest N of the run's checkpoints, counting the ones an earlier,
  interrupted process of the same run wrote. It never deletes the checkpoint a run was resumed from, nor
  anything in the directory that another run wrote. Delete the checkpoint directory once you have chosen a
  model.
- **Where.** The checkpoint directory must be outside `--out`, and is created and checked before training
  starts, so a place that cannot be written to is an error in the first second and not at the first checkpoint.
- **Durability.** A checkpoint, and the final model, is written to a sibling directory, flushed to disk, and
  renamed into place. A directory with a checkpoint's name is a complete checkpoint, after a killed process or a
  power cut alike. The final model is written as soon as training ends, before the results are measured.
- **The model a run started from must stay where it is until the run ends.** Each save copies its tokenizer and
  configuration files from there.

**How long it takes.** Each step encodes `2 x batch-size` texts and its cost follows their length. Measured on an
Apple-silicon laptop with the default batch size of 32:

| Texts | CPU | Metal |
|---|---|---|
| Names (about 10 tokens) | 0.3 s per step, 0.9 GB of memory | 0.2 s per step |
| Descriptions at the 256-token limit | 9 s per step, 9 GB of memory | 1.3 s per step |

A run has `examples / batch-size x epochs` steps, so 10,000 description examples for two passes is about 625 steps:
around an hour and a half on that CPU, or a quarter of an hour on its GPU. Use a GPU, or a smaller `--batch-size`,
when the CPU's memory use is a problem.

### 3. Choose a model, then measure it once

Two different questions, and they need different pairs. *Which* model (which checkpoint, which learning rate,
which threshold) is a choice, and whatever pairs make it have been fitted to. *How good is it* is a measurement,
and only pairs that chose nothing can make it. An export has one split nobody trains on, the holdout, so it can
answer only one of the two. Give the choosing a split of its own:

```sh
record-similarity pairs carve pairs.full.jsonl --fraction 0.15 > pairs.carved.jsonl
```

That moves a share of the `calibrate` owners to a new `validation` split. Whole owners move, never single pairs,
and which ones is decided by a hash of the owner's name and `--salt`, so the same command always moves the same
owners. Pick the share so that the new split has enough of both labels to tell models apart; the command logs
how many edits and recycles it moved, and warns if either is none. Catalog groups whose `owner` is in the
validation or holdout split are left out of training too, so that a brand held back for measuring is not learned
from by another route.

Train on the carved file and the trainer does the choosing. It scores the model it started from, every
checkpoint, and the final model on the validation pairs, and names the best in its `finished` line
(`best_model`, `best_step`). The score is the **AUC**: how often an edit scores above a recycle, which needs no
threshold. That matters, because the configured thresholds belong to the base model and would judge every other
model on the wrong scale. `--out` still holds the final model; if a checkpoint or the base did better, the line
says which, and `--keep-checkpoints` never deletes the best one. On the bundled sample the base model already
separates every validation pair, so three passes of training cannot beat it and the run says so:

```text
validation before     step   0  auc 1.0  models/mdbr-leaf-mt
validation checkpoint step   5  auc 1.0  ft.checkpoints/step-000005
validation checkpoint step  10  auc 1.0  ft.checkpoints/step-000010
validation after      step  15  auc 1.0  ft
held_out   selected   step   0  auc 1.0  models/mdbr-leaf-mt
```

(The `separation` lines of that run at width 256, reduced to five of their fields. A tie goes to the earlier
model: the same result for less training.)

For the full policy on a candidate, with field weights, floors, and quantities, calibrate it on the validation
split. `--model` points at any model directory or checkpoint without a second config file:

```sh
just calibrate pairs.carved.jsonl config.toml --full --split validation --model models/ft.checkpoints/step-000120
```

Its report has the same AUC over the overall similarity, the suggested threshold with its margin, and the
operating points. Set the new model's thresholds from this.

Then the holdout, once, for the model that was chosen, against the base model:

```sh
just calibrate pairs.carved.jsonl config.toml --full --split holdout                               # base model
just calibrate pairs.carved.jsonl config.toml --full --split holdout --model models/mdbr-leaf-mt-ft
```

Keep the fine-tuned model only if it is clearly better there. A model that improves only on the pairs it trained
on has memorized them. The trainer follows the same rule: with validation pairs, the only held-out line it
prints after training is for the model that was chosen.

### 4. Deploy it

A fine-tuned model is a different model. Its fingerprint differs, so:

1. Set `model.dir` to the new directory and `qdrant.collection` to a **new** collection name.
2. Set the thresholds from the calibration of the new model on the `validation` split (step 3). The old
   thresholds do not carry over. If the holdout is all there is, use it, and expect production to be somewhat
   worse than the error rates it then reports.
3. Re-index: see [Re-indexing](#re-indexing).
4. To ship it in the container image, `COPY` the directory into the image in place of the `model fetch` step in
   the `Dockerfile`, and remove `/models` from `.dockerignore` so the build can see it.

The saved directory is also a valid sentence-transformers model. If you want a second opinion from PyTorch,
`just verify-model models/mdbr-leaf-mt-ft` (needs `uv`) loads it there and requires the Burn encoder to reproduce
PyTorch's embeddings. Nothing in the workflow depends on it.

### Where a model came from

Every model directory these tools write holds a `provenance.json` saying what produced it. After a few rounds of
fine-tuning that is the only way to tell which base, data, and options made `models/mdbr-leaf-mt-ft`, and
thresholds belong to one specific model. This is from a one-pass run on the bundled sample (shortened):

```json
{
  "format": 1,
  "written_at": 1794378917,
  "tool": "record-similarity-train",
  "version": "0.1.0",
  "commit": "38175fdd5786",
  "dirty": true,
  "digest": "a3dde74c22049b9d",
  "trained": {
    "base": {
      "dir": "models/mdbr-leaf-mt",
      "digest": "f46e48ee0ea2dd02",
      "provenance": {
        "tool": "record-similarity model fetch",
        "digest": "f46e48ee0ea2dd02",
        "fetched": {"repo": "MongoDB/mdbr-leaf-mt", "revision": "1ed41b22ce166d66c24f88ebfc340e1f03adb20f"}
      }
    },
    "data": [{"role": "pairs", "path": "data/calibration.sample.full.jsonl", "sha256": "a65fc350...", "examples": 29}],
    "data_fingerprint": "e71757b4cee8e26fd7a5b77432302746",
    "options": {"epochs": 1, "learning_rate": 0.00002, "seed": 0},
    "max_tokens": 256,
    "device": "cpu",
    "resumed": [],
    "step": 1,
    "total_steps": 1
  }
}
```

- **`digest`** is the first part of the model's fingerprint (`a3dde74c22049b9d-d256-t256`): the files the encoder
  is built from, and nothing else. The server logs it, and what it finds in this file, on its `model loaded` line.
- **`trained.base.provenance`** is the base model's own file, nested, so the chain back to the published model is
  in one place. A base without one is trained from all the same; the record then starts with its digest.
- **`data`** names each file and the SHA-256 of its text. `data_fingerprint` is over the examples themselves, in
  order, and is what a resumed run is checked against.
- **A checkpoint has the same file**, with `step` below `total_steps`. A resumed run keeps the base it first
  started from and lists each checkpoint it was continued from under `resumed`.
- **`commit` and `dirty`** are the source the binary was built from, and whether that tree had uncommitted
  changes. A build without git (the Docker image, unless `RECORD_SIMILARITY_COMMIT` is passed in, as `just image`
  does) records `unknown`.
- **`model fetch` writes one too**, with the repository, the revision, and the checksum of every file. Running
  `just fetch-model` again adds it to a model that was fetched before this existed.

The file records what produced the model, never what the model scored. It is not part of the model either: adding
it, editing it, or deleting it changes no fingerprint, and nothing refuses a model for not having one.

### Recorded runs

A fine-tuning run and a calibration each leave a directory under `runs/`, without being asked:

```text
runs/20261111T201826Z-train-mdbr-leaf-mt-ft/
  run.json        what ran: the build, the machine, the command line, the options, the model, the data, how it ended
  events.ndjson   everything the run emitted, in order: every result and failure, and logs at `info`
  report.txt      for a calibration read as text, the report
```

Nothing is sent anywhere, and the directory is the whole record. It exists so that the answer to "which options
gave that result" is a file and not somebody's shell history:

```text
$ just runs
20261111T201826Z-train-mdbr-leaf-mt-ft               train      finished    16.0s  f46e48ee0ea2dd02             data/calibration.sample.jsonl
20261111T201704Z-calibrate-calibration-sample-full   calibrate  finished     1.1s  f46e48ee0ea2dd02-d256-t256   data/calibration.sample.full.jsonl
20261111T201701Z-calibrate-calibration-sample        calibrate  finished     2.5s  f46e48ee0ea2dd02-d256-t256   data/calibration.sample.jsonl
```

Each line is the run's id, its kind, how it stands, how long it took, the model (the base it trained from, or the
fingerprint it scored with), and its first input.

- **`events.ndjson` is the run's output, kept.** For training it has the `training` lines the loss curve is made
  of, whatever `RUST_LOG` was: quietening stderr does not thin the record. For a calibration it has the events
  `--ndjson` writes, with or without `--ndjson`, so any two calibrations can be compared later:
  `just calibrate-diff runs/<one>/events.ndjson runs/<other>/events.ndjson`.
- **A run that was killed says so.** A manifest is written as `running` and rewritten when the run ends. A
  process that is killed writes nothing, so `runs list` works it out: a run still marked `running` whose process
  no longer holds its lock is shown as `killed`.
- **A resumed run is a run of its own.** `--resume` starts a new directory whose `run.json` names the checkpoint
  it went on from and the run that wrote it (`continues`). The model's `provenance.json` names the run that wrote
  it (`run`), and each checkpoint it was resumed from names its own.
- **`--runs DIR`** (or `RECORD_SIMILARITY_RUNS`) puts the records somewhere else, and **`--no-run`** records
  nothing. If the directory cannot be written, the command stops before doing any work and says so.
- **Runs stay where they were made.** `runs/` is ignored by git and left out of the container image. A run on
  real data names its files and holds pair ids and owner pseudonyms in its events; `run.json` also has the host
  name, the working directory, and the command line.

Only `calibrate` and the trainer are recorded. `serve`, `import`, `compare`, and `calibrate-diff` are not.

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
| `policy.reject_on_measure_change` | `false` | Reject a revision whose measures changed: a quantity that became incomparable with its old value, or one quantity removed as another appeared. Reported as `measure_changed` either way. |
| `fields.<name>.weight` | (required) | Relative importance of the field. Only ratios matter. |
| `fields.<name>.required` | `false` | Refuse records with no value for this field. |
| `fields.<name>.reject_below` | none | Reject a revision whose similarity for this field alone is below this. |
| `[quantities.<name>]` | none defined | Naming the table defines the quantity, even with nothing in it. It is then accepted in requests, compared, and reported. |
| `quantities.<name>.reject_above_factor` | none | Reject a revision that changes this quantity by more than this factor, in either direction, or moves it between zero and a value above zero. At least 1. |
| `quantities.<name>.reject_incomparable` | `false` | Reject a revision that puts this quantity in a unit that cannot be compared with the one it had. |

At least one field is required. Field and quantity names may contain letters, digits, and underscores, and a name
cannot be both. The example configuration defines four quantities and sets no limit on any of them.

What a change costs:

| Change | Consequence |
|---|---|
| A threshold, a floor, `required` | None. Takes effect on restart. |
| A quantity added, removed, or given a limit | None. Quantities are stored beside the vectors, not in them. A record stored before a quantity existed reports it as `ADDED` the first time it arrives. |
| A field weight | Verdicts use the new weight at once. Search rankings use the weights records were written with until they are rewritten. |
| `model.dir`, `output_dim`, `max_tokens`, or the set of fields | Stored vectors are no longer comparable. Use a new collection and re-index. The server refuses to mix them. |

## Operations

### Importing a catalog

`record-similarity import` loads an export straight into the store, a batch at a time, without going through the
gRPC API. It is how a catalog is loaded for the first time and how it is loaded again after a change of model.

```sh
just import export.jsonl config.toml --check     # only validate every line against the schema: no model, no Qdrant
just import export.jsonl config.toml             # import into the collection the config names
```

The export is one JSON object per line. `record` is the record as it is now, in the shape every other tool reads
(a string is a text field, an object is a quantity). `tenant` and `anchor` are optional:

```json
{"id": "sku-1", "tenant": "brand-7", "record": {"name": "Blueberry Gummies 100mg - Indica", "category": "Edibles > Gummies", "weight_per_item": {"value": 40, "unit": "g"}}, "anchor": {"name": "Blueberry Gummies 100mg", "category": "Edibles > Gummies"}}
```

What it does, and what it will not do:

- **It only creates.** An id that is already stored is skipped and left exactly as it is. An import can therefore
  never overwrite a baseline, running one twice is harmless, and an import that was interrupted is resumed by
  running it again.
- **`anchor` is the record as it was first created**, if your database still has that version. With it, the
  record's drift is measured from where it really began, as if it had lived here all along: the anchor holds the
  original value of every field and quantity the original had, and today's value for anything it lacked. Without
  it, the record as imported is its own point of reference, as for any new record.
- **A record that cannot be imported fails by itself.** It is logged with its line, its id, and the reason, the
  rest of the export goes in, and the command ends with a `failed` line and a non-zero exit status if there was
  any. `--stop-on-error` stops at the first one instead. `--check` finds the same problems in a second or two,
  before anything is embedded.
- **It needs the model and a connection to Qdrant**, like the server, and reads the same config. It does not
  need the server. Run it before the server takes edits for the records being imported: like two replicas
  ([Limitations](#limitations)), an import and a server that both see a new id at the same instant do not
  coordinate, and whichever writes second wins.

Progress is logged after every batch (`--batch-records`, 64 by default), and the outcome is a `finished` result
on stdout. This is an export of 48 synthetic records, two of them flawed, with the envelope fields removed:

```json
{"level":"INFO","message":"importing","records":32,"imported":32,"skipped":0,"failed":0,"elapsed_s":1.2}
{"level":"WARN","message":"record not imported","line":49,"id":"p900","reason":"invalid record: `weight_per_item` is a quantity, not a text field; give it as a number with a unit"}
{"level":"WARN","message":"record not imported","line":50,"id":"p901","reason":"invalid record: required field `name` is missing or blank"}
{"level":"INFO","message":"finished","checked_only":false,"records":48,"imported":46,"skipped":0,"failed":2,"elapsed_s":1.3}
```

Running it again imports nothing and skips the 46. One of those records was imported with the version it was
created as. Asked about its current content, the server answers that nothing has changed since the import, and
that the record has already moved from where it started:

```text
$ just client evaluate p026 p026.json --tenant brand-1 --dry-run
verdict   Accepted (stored: false, revision 1)
overall   1.000 against a threshold of 0.900
drift     0.963 similarity to the original record
```

### Re-indexing

Needed after changing the model, `output_dim`, `max_tokens`, or the set of fields. Stored vectors carry an identifier
of the model and field layout that produced them, and the server will not compare across layouts: reading an old
record returns `FAILED_PRECONDITION`, and a collection whose vectors have the wrong width stops the server at
start-up.

1. Point `qdrant.collection` at a new, empty collection.
2. Export every record from your own database and [import](#importing-a-catalog) it. The service stores vectors
   and never the text, so the text has to come from you.
3. Restart the server on the new collection, and drop the old one when you are satisfied.

**Drift history survives a re-index only if the export carries it.** Give each record's original version as
`anchor` and its drift is measured from there, as before. Leave it out and each record's anchor becomes its
content at the time of the import.

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
just test-snapshot   # the curated hard cases against the pinned verdicts, and the README's quoted reports
just test-real       # both of those, then the full stack through gRPC against Qdrant
just snapshot        # rewrite the pinned verdicts after an intended change, printing what moved
just throughput      # forward-pass latency on this machine
```

Run `just test-real` after touching the encoder, the loader, the tokenizer settings, the trainer, or
`config.example.toml`.

**Pinned verdicts.** `data/regression.jsonl` holds two dozen synthetic pairs chosen to be hard: the same amount
in another unit, a same-brand variant, a field blanked and refilled with another product's text, look-alike
characters, an edit past the token limit. `data/regression.snapshot.ndjson` is what the pinned model and
`config.example.toml` make of each, as a `calibrate --ndjson` run. Two tests hold it:

- `just test` judges the snapshot again under the current `config.example.toml` and scoring code, with no model.
  A changed weight, floor, threshold, limit, or judging rule that moves any case fails there, in a fraction of a
  second.
- `just test-snapshot` scores the cases afresh with the real model, which is what catches a change to the encoder,
  the loader, or the tokenizer. It also checks that the calibration reports quoted under
  [Choosing thresholds](#choosing-thresholds) are what `calibrate` prints. CI runs it.

When a change is meant to move a case, `just snapshot` prints what moved and rewrites the file; that printout
belongs in the commit message. The cases pin behavior, not correctness (several are `unsure` on purpose), and no
threshold may be read from them: they were picked, not sampled.

**Property tests.** `just test` includes tests that state a rule and let the machine look for an input that
breaks it (the `properties` modules of `similarity-core` and `similarity-server`, on
[proptest](https://proptest-rs.github.io/proptest/)). The largest drives the engine with random sequences of
requests, in every write mode, and after each one checks that a rejected revision wrote nothing, that the anchor
lost nothing it held, that the revision counter moved by exactly one write, and that no request touched a record
it did not name. Each property runs 64 cases by default so that the suite stays fast;
`PROPTEST_CASES=5000 just test` is the thorough run. A failure is shrunk to a small input and saved under the
crate's `proptest-regressions/`, which is checked in, so the case is replayed first from then on.

### Project layout

```text
crates/
  similarity-log/       logging for both binaries: one JSON object per line, nothing else
  similarity-proto/     the .proto file and generated gRPC code
  similarity-core/      the encoder (Burn), tokenization, composite vectors, quantities, scoring, policy. No network, no async.
  similarity-store/     the VectorStore trait, an in-memory implementation, and the Qdrant one
  similarity-server/    config, embedding worker thread, engine, gRPC service, import, calibration, and the `record-similarity` binary
  similarity-train/     fine-tuning: data, losses, the training loop, and the `record-similarity-train` binary
data/
  calibration.sample.jsonl        synthetic labeled pairs, in the strict format
  calibration.sample.full.jsonl   the same pairs in the full format a data export uses
  regression.jsonl                curated hard cases, synthetic, whose verdicts are pinned
  regression.snapshot.ndjson      what the pinned model and the example config make of them
  examples/                  sample records for the client
docs/
  decisions.md               why things are the way they are (D1 to D24)
  data-request-prompt.md     a brief for collecting labeled production data
  feature-suggestions.md     twenty features that could be added, none built yet
  plan-features.md           a plan for the first five of them
models/                 fetched and fine-tuned model directories (ignored by git)
tools/                  referees, not part of any pipeline: they write the fixtures the parity tests compare with
  golden.py             PyTorch reference embeddings
  train_golden.py       PyTorch reference training steps
```

Conventions (two-space indentation, no `unwrap` in library code, tests first) are in [`CLAUDE.md`](CLAUDE.md).

## Troubleshooting

An error from either binary arrives as the `error` field of a `failed` line on stderr; the messages below are
what that field says.

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
| `... already holds a checkpoint for step N, which this run would write again` | The checkpoint directory has checkpoints from an earlier run of the same name. Resume from the latest with `--resume`, or delete them, or give another `--checkpoint-dir`. |
| `checkpoint ...: belongs to a run with different options` (or `different training data`) | `--resume` was given with other options or data than the run was started with. Repeat the original command, changing only `--base` to `--resume`. |
| `INVALID_ARGUMENT: invalid record: unknown field ...` | The request used a field name the schema does not define. `Describe` lists the valid ones. |
| `INVALID_ARGUMENT: invalid record: unknown quantity ...` | The request used a quantity the schema does not define. Add a `[quantities.<name>]` table to the config, or drop it from the request. |
| `` invalid record: `...` is a quantity, not a text field `` | A quantity was sent as text in `fields`. Send it in `quantities`, as a number with a unit. |
| A quantity reports `INCOMPARABLE` for a unit that should convert | The unit is not in the service's table, so it is comparable only with itself. The table is `UNITS` in `crates/similarity-core/src/quantity.rs`; adding a unit is one line. |
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
- **No bulk RPC.** A catalog is loaded with the `import` command, which has to run where it can reach the model
  and Qdrant. Over gRPC it is still one `Evaluate` call per record.
- **English only.** The default model is English. Other languages need another model and a new calibration.
- **Text and quantities only.** Images are not considered. A recycler who changes only the photo is invisible to
  this service.
- **No drift check for quantities.** A factor limit applies to each revision against the one before it. A quantity
  can still be moved a long way in steps that each stay under the limit; the text drift check
  (`anchor_threshold`) does not see quantities.
- **Quantities play no part in search.** `FindSimilar` compares vectors, which hold text only.
- **Truncation.** Text beyond `max_tokens` (256 by default) is ignored. A long field is tokenized a part at a
  time and no further than it takes to fill that limit, so an enormous description costs little more than an
  ordinary one (D21). The whole text still decides whether a field counts as changed.
- **No transport security or authentication** in the server itself.

## Acknowledgements

- [Burn](https://github.com/tracel-ai/burn) for the tensor and module framework.
- [`MongoDB/mdbr-leaf-mt`](https://huggingface.co/MongoDB/mdbr-leaf-mt) (Apache-2.0), the default embedding model.
- [sentence-transformers](https://www.sbert.net), whose model directory layout this project reads and writes, and
  whose output is the reference both parity tests are held to.
