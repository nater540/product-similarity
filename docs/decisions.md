# Decisions

The choices that shaped this project, with the evidence behind them. Code comments refer to these by number.
Numbers quoted here were measured on an Apple-silicon laptop CPU with a release build; rerun `just throughput` and
`just calibrate` to reproduce them.

## D1. Burn 0.22 on the `flex` CPU backend, Metal behind a feature

Burn 0.22 changed how backends work: models are plain types, and a runtime `Device` selects the backend. `ndarray`
and `tch` are deprecated; `flex` is the pure-Rust CPU backend that replaces them. The workspace builds `flex` (with
`simd` and `rayon`) always, and `similarity-server` adds Metal with `--features metal` for development on a Mac.

The server is built with no training features (`default-features = false`): no optimizer and no autodiff, and the
encoder has no dropout layer in its module tree. Leaving those out of the serving path removes a class of "why is
this embedding different today" bugs. Training exists, but in its own crate and its own binary: see D13.

Burn 0.23 was in pre-release when this was written; 0.22.1 is the latest stable.

## D2. A BERT module tree that mirrors Hugging Face's, loaded strictly

`similarity-core::model::bert` defines the encoder with the same structure and field names as Hugging Face's
`BertModel` state dict, rather than using Burn's stock `TransformerEncoder`. The key-remapping table is three
entries long as a result, and a reader can match checkpoint tensor names to structs by eye.

Loading is audited (`model::loader`): no tensor may fail, no parameter may be left without a value, no checkpoint
tensor may go unused except a short named list (`pooler.`, `cls.`, `embeddings.position_ids`), and the number of
tensors applied must equal the number the config implies. A half-applied checkpoint would still produce unit
vectors and confident verdicts, which is exactly why it must be impossible.

Every checkpoint is cast to `f32` on the way in. Loaded parameters keep the checkpoint's dtype, and `gte-small` is
published in `f16`: without the cast it loaded cleanly and then panicked in the backend on the first forward pass.
`Embedder::load` also runs one real forward pass before returning, so anything of that kind fails at start-up.

The pipeline around the encoder (pooling, optional projection) is read from the model directory, in both formats
sentence-transformers has used for `1_Pooling/config.json`: one boolean per mode (what most Hub repositories
contain) and a single `pooling_mode` name (what release 5.7 writes when it saves a model, for example after
fine-tuning).

What counts as a supported pipeline is defined once, in `model::sentence::Pipeline`: a `Transformer`, one
`Pooling`, optionally one `Dense`, optionally a final `Normalize`. The loader builds from it and `model fetch`
derives its download list from it, so a model is refused before its weights are downloaded exactly when it would
be refused at load. Two rules in it exist because of what `modules.json` is, a file that arrives with the model:

- A stage's path must be a subdirectory of the model directory. It is joined onto that directory for reading,
  downloading, and saving, and `../..` would take all three outside it.
- `Normalize` is accepted only as the last stage. There it is a no-op, since every embedding is normalized on the
  way out. Before a `Dense` it is not: PyTorch computes `W(x/|x|) + b`, and skipping it would compute `Wx + b`,
  which is a different direction, with nothing to show for it.

**Proof that the port is right:** `tests/real_parity.rs` compares the Burn output on the real checkpoint with what
the reference PyTorch pipeline produces (`tools/golden.py`). Across the ten fixture texts the cosine between the two
is above 0.99999 and the largest per-component difference is below 2e-4.

## D3. The model: `MongoDB/mdbr-leaf-mt`

The requirement was "lightweight but accurate". The candidates worth considering at this size are all the same
architecture (BERT, 384 hidden, 12 heads) and differ in depth, pooling, and training, so one implementation loads
any of them from a sentence-transformers directory and the choice is a configuration value.

What decides it is how well each separates ordinary edits from recycled products on `data/calibration.sample.jsonl`
(26 edits, 20 recycles; four fields, weighted as in `config.example.toml`). "Gap" is the distance between the
lowest-scoring edit and the highest-scoring recycle: the room a threshold has to sit in.

| Model | Width used | Layers | Params | Edits | Recycles | Gap | One record |
|---|---|---|---|---|---|---|---|
| **`mdbr-leaf-mt`** | 256 of 1024 | 6 | 23M | 0.941 to 1.000 | 0.495 to 0.850 | **0.091** | 20 ms |
| `mdbr-leaf-mt` | 1024 | 6 | 23M | 0.946 to 1.000 | 0.547 to 0.886 | 0.060 | 20 ms |
| `all-MiniLM-L6-v2` | 384 | 6 | 22M | 0.896 to 1.000 | 0.173 to 0.815 | 0.081 | 19 ms |
| `gte-small` | 384 | 12 | 33M | 0.968 to 1.000 | 0.791 to 0.949 | 0.019 | 41 ms |
| `bge-small-en-v1.5` | 384 | 12 | 33M | 0.928 to 1.000 | 0.583 to 0.911 | 0.017 | 42 ms |

What this supports, and what it does not:

- **The two 12-layer models are the wrong tool.** They are twice as slow and leave almost no room for a threshold.
  They are tuned for retrieval, where "blueberry gummies" and "peach sour belts" *should* be close.
- **`mdbr-leaf-mt` and `all-MiniLM-L6-v2` are close, and 46 synthetic pairs cannot separate them.** `mdbr-leaf-mt`
  at 256 dimensions has the slightly wider overall gap; `all-MiniLM-L6-v2` separates the `name` field alone more
  widely (0.780 against 0.562, where `mdbr-leaf-mt` has 0.875 against 0.725).

`mdbr-leaf-mt` is the default because its model card reports the best MTEB (English, v2) score among models under
30M parameters, it is Apache-2.0 licensed, and it can be truncated (D4), which `all-MiniLM-L6-v2` cannot. That is a
reasonable prior, not a measured win. When labeled production pairs exist, run `calibrate` with both and keep
whichever has the wider gap; switching is a config value, a new collection, and a re-index.

Two consequences worth remembering:

- **Similarities are not percentages.** With this model two unrelated products score about 0.5, not 0. A threshold
  is only meaningful for the model it was calibrated on.
- **The calibration pairs are synthetic.** They were written for this repository to be plausible, not sampled from
  production. The ranking of models is probably robust; the exact threshold is not. Re-run `calibrate` on labeled
  production edits before enforcing.

English only. A multilingual catalog needs a multilingual model and a new calibration.

## D4. 256 dimensions per field, as an explicit setting

`mdbr-leaf-mt` is trained with Matryoshka representation learning: a prefix of its 1024 output components is itself
a usable embedding. On the calibration set the gap between edits and recycles does not shrink as the width does:

| Width | Edits from | Recycles up to | Gap |
|---|---|---|---|
| 1024 (native) | 0.946 | 0.886 | 0.060 |
| 512 | 0.939 | 0.868 | 0.071 |
| 256 | 0.941 | 0.850 | 0.091 |
| 128 | 0.948 | 0.841 | 0.107 |

The trend toward a wider gap at smaller widths is probably an accident of 46 pairs and should not be read as
"smaller is better". What the table does show is that truncation costs nothing measurable here, so
`config.example.toml` sets `output_dim = 256`: a four-field composite is then 1024 floats (4 KB) instead of 16 KB.

`output_dim` has **no default**: leaving it out keeps the model's native width. Truncating a model that was not
trained for it (`all-MiniLM-L6-v2`, `gte-small`, `bge-small-en-v1.5`) throws away part of the embedding, so it must
be something a configuration asks for, never something it inherits. (An earlier revision defaulted to 256, and the
first version of the table in D3 was silently measured with every model truncated as a result.)

`output_dim` is part of the model fingerprint, so changing it means a new collection.

## D5. Field-aware scoring through one composite vector

A record is embedded field by field. The per-field unit vectors are scaled by the square root of their weight,
concatenated, and normalized into one **composite** vector, with zeros where a field has no value.

- The cosine of two composites equals the weighted mean of the per-field cosines. So the score Qdrant computes
  during a search is the same number the accept/reject policy uses, with a single vector and a single index.
- Cosine ignores scale, so per-field similarities can be recovered from any two composites by slicing them into
  blocks, even composites written under different weights. Changing a weight never invalidates stored comparisons.

Field-aware scoring is what makes the policy work: a recycler who swaps the name and leaves a long description
alone moves a single-vector embedding very little, but moves the name block a lot. Each field can carry its own
`reject_below` floor for that reason.

Fields that are added or removed are reported but left out of the mean. Filling in an empty description is
maintenance, not evidence of a different product; a field that must not disappear should be `required`.

**Where search and verdicts part ways.** The identity "cosine of composites equals the weighted mean" holds when
both records have values for the same fields. When they do not, `Evaluate` leaves the unshared fields out, but a
composite has zeros there, so the search score comes out lower: two records identical except that one has no
description score 1.0 in `Evaluate` and about 0.89 in `FindSimilar` with the example weights. Verdicts are
unaffected. Search rankings are skewed toward records with the same set of filled-in fields as the query, which
matters most for ad hoc queries that supply only a name. The fix, if it matters in practice, is to over-fetch
candidates and re-rank them with the exact weighted score.

## D6. What Qdrant is, and is not, doing here

The honest assessment, since the question was asked:

- **The core operation does not need a vector database.** Evaluating a revision is a point lookup by id followed by
  a few dot products in the server. No nearest-neighbor search is involved.
- **At tens of thousands of records, search does not need one either.** Brute force over 50,000 composites of 1024
  floats is about 50 million multiply-adds, on the order of 10 ms on one core. `MemoryStore` does exactly that.
- **What Qdrant does provide** is everything operational: durable state shared by every server replica,
  tenant-filtered search, snapshots, a dashboard for looking at what is stored, and headroom to millions of records
  without redesign.
- **What it lacks** is a compare-and-swap. Two replicas evaluating the same id at the same instant can both read
  the same baseline and both write (see D10). A relational database would close that with a row lock or a version
  check.

So: Qdrant is a reasonable choice because neighbor search is wanted and it is pleasant to operate, not because the
scale demands it. If the organization already runs Postgres, a table with a `bytea` or `pgvector` column would do
the same job at this scale with one fewer service and with transactional writes. The store is behind a trait
(`VectorStore`, four methods, one contract test suite), so that swap is a few hundred lines.

Layout in Qdrant: one collection, one point per record, two named vectors. `current` is HNSW-indexed and is what
searches match. `anchor` (see D8) is only ever read by id, so its HNSW graph is disabled (`m = 0`) and it lives on
disk. `tenant` has a tenant-aware keyword index. Point ids are the UUIDv5 of the record id.

The payload indexes are ensured on every connect, not only when the collection is created. Creating an index
that exists costs nothing, and the alternative leaves a collection whose first start failed between the two steps
without indexes permanently: filters still work unindexed, so nothing would ever report it.

## D7. The server decides, and a rejected revision changes nothing

`Evaluate` embeds a record and, if it has an id, judges it against the stored baseline:

| Situation | Verdict | Written? |
|---|---|---|
| No id | `EMBEDDED` | never |
| Id, no baseline | `CREATED` | yes |
| Id, baseline, passes | `ACCEPTED` | yes, unless nothing changed |
| Id, baseline, fails | `REJECTED` | no |

A rejected revision must leave the baseline alone. If it overwrote it, the recycler's second attempt would be
compared against their own first attempt and pass.

Three write modes cover the workflows around that: `ENFORCE` (above), `DRY_RUN` (never write: preview a verdict
while the merchant is still typing), and `FORCE` (always write: a human reviewer overrides the policy). A forced
write still reports the verdict the policy would have given.

A `mode` value the server does not recognize is `INVALID_ARGUMENT`. Protobuf's convention is to read an unknown
enum value as the default, and the default here writes: a newer client asking an older server for some future
non-writing mode would have its record stored.

## D8. An anchor, so that small steps cannot walk a record anywhere

Every accepted edit moves the baseline. A recycler who learns the threshold can turn gummies into a vape cartridge
in six edits that each look fine (the engine's test suite does exactly this). So each record also stores an
**anchor**: its composite as first stored. Every response reports `anchor_similarity`, and `anchor_threshold`
optionally rejects on it. A `FORCE` write resets the anchor, since a reviewer has said "this is the record now".

**The anchor also closes a two-step swap, whether or not drift is enforced.** A field that only one version has
cannot be compared, so it is left out of the score: adding a description that never existed is maintenance, not
recycling. Taken alone, that rule let any optional field be replaced unseen, by blanking it in one edit and
filling it with something else in the next; neither edit compared the new text with the old. Two additions fix
it:

- A field the baseline lacks but the anchor has is reported as `RESTORED` and scored against the anchor's value,
  in the overall score and against the field's floor. The refill is judged as the direct change would have been.
- When an accepted revision adds a field the anchor has no value for, the anchor takes it. So the anchor holds the
  *first* value of every field the record has ever had, and a field added later is protected the same way (and
  now takes part in drift, which it did not before).

Values the anchor already holds are never replaced by an ordinary write; that part of the rule is unchanged. The
comparison is with the field's first value rather than its last before blanking, which would need a third stored
vector. The two differ only for a field that had drifted legitimately before being blanked and restored, and
there the drift check was already making the same comparison.

## D9. One embedding thread, batching by backlog

Inference is synchronous CPU work and must not run on a tokio worker. `EmbedWorker` owns one OS thread that owns
the model. One thread and not a pool, because the backend already spreads a forward pass across cores.

While a pass runs, requests queue; when it finishes, the worker takes everything waiting and runs it as one batch.
A lone request waits for nothing, and batches grow by themselves under load, with no timer to tune. The queue is
bounded and a full queue is `RESOURCE_EXHAUSTED` immediately.

Within a batch, texts are bucketed by length, because every row is padded to the longest and cost follows the
padded size. That took a typical four-field record from 31 ms to 20 ms.

Merging requests must not merge their failures. If the model fails on a merged batch (an input it cannot embed, a
panic in the backend), each request of the batch is embedded again on its own, so the error reaches the request
that caused it and the others get their vectors. The retry runs only on a path that was already failing.

Fields whose text has not changed are never sent to the model at all: each stored record keeps a SHA-256
fingerprint of each field's (whitespace-normalized) text, and an unchanged field reuses its stored block. A typical
edit touches one field, so a typical request embeds one short text, about 4 ms.

## D10. Per-id locking is in-process

Evaluating is read, compare, write. `Engine` serializes requests for the same id with striped locks, so within one
server two concurrent revisions cannot both pass against the same baseline. Across replicas there is no such
guarantee (see D6). This is acceptable when the calling system already serializes edits to one record, which a
product-editing UI generally does. If it does not, route by id to one replica or move the store to something with
compare-and-swap.

## D11. The server takes paths; only `model fetch` touches the network

`Embedder::load` takes a directory. The serving path contains no HTTP client and starts identically in an
air-gapped cluster. `record-similarity model fetch` is the single place a URL appears. It pins a commit hash, and
for the default model the SHA-256 of every file is compiled in. The Docker image runs it at build time, so every
replica has the same weights.

Each repository is fetched into its own directory (`models/<repository name>` unless `--dir` says otherwise).
With one fixed default, fetching a second model replaced the first one's files in place, and the directory then
loaded without complaint under a configuration and thresholds calibrated for the model it used to hold.

The offline reports (`compare`, `calibrate`) print the model fingerprint and the width they ran at as their first
line. That information decides whether two reports can be compared at all (D4), so it belongs in the report and
not in a log line that a `RUST_LOG` setting can hide.

## D12. Stored vectors carry their layout, and layouts never mix

A stored composite is only meaningful to a server with the same model (including `output_dim` and `max_tokens`) and
the same set of field names: otherwise its blocks would be sliced at the wrong offsets, or compared against vectors
from a different model, and the result would be wrong without looking wrong.

`Schema::layout_id` names the layout (model fingerprint plus field names). Every point stores it, searches filter
on it, and reading a record under another layout is a `FAILED_PRECONDITION` that says to re-index. At start-up the
server also refuses a collection whose vector width does not match.

In practice: changing a **weight** or a **threshold** needs nothing. Changing the **model**, `output_dim`,
`max_tokens`, or the **set of fields** needs a new, empty collection and a re-index (call `Evaluate` once for each
record; every one becomes a fresh baseline).

## D13. Fine-tuning in Burn too, in a crate the server does not link

Fine-tuning first lived in a PyTorch script, on the argument that the recipe already existed there. It was moved
into Rust (`similarity-train`, binary `record-similarity-train`) so that the project has one implementation of the
model instead of two that must be kept in agreement: the trainer runs the same `SentenceEncoder`, the same
tokenizer wrapper, and the same safetensors loader and saver as the server.

**The server stays inference-only.** `similarity-train` is the only crate that enables Burn's `autodiff` and
`optim` features. Cargo unifies features per build, so `cargo build -p similarity-server` (what `just release` and
the `Dockerfile` run) links neither; `cargo tree -p similarity-server -e features` shows no `burn-autodiff`. A
workspace-wide `cargo test` does compile them in, which changes nothing: gradients are only tracked for a module
that has been put into training mode.

**Dropout is an argument, not a layer.** BERT is fine-tuned with dropout, and D1 keeps it out of the module tree.
`BertModel::forward_training` takes a `TrainingDropout` and applies it where BERT does (embeddings, attention
weights, each sub-layer's output); `forward` has no code path that could apply one. There is no mode flag on the
model for a caller to forget.

**Proof that the trainer is right.** A training bug does not crash, it produces a quietly worse model, so the
trainer is held to PyTorch the same way the encoder is (D2). `tools/train_golden.py` takes six optimizer steps in
PyTorch on fixed batches with dropout off and records each step's loss and global gradient norm and the embeddings
of some probe texts afterwards. `similarity-train/tests/real_parity.rs` replays them: losses agree to six decimal
places, gradient norms to about one part in 100,000, and the probe embeddings move in the same direction by the
same amount (cosine of the displacement 1.0000). That covers the training-mode forward pass, both losses, the whole
backward pass, the clipping (two of the six steps are clipped), and the optimizer. Dropout itself is outside it,
being random; unit tests check that it perturbs training passes and never inference.

**The recipe**, and where it deliberately differs from the library defaults it replaces:

- `AdamW` with epsilon 1e-8 and no weight decay, linear warm-up over 10% of the steps and linear decay after, as
  sentence-transformers trains. Burn's own `AdamW` defaults (epsilon 1e-5, decay 1e-4) are overridden.
- Gradients are clipped to a **global** norm of 1 by a visitor over the module. Burn's built-in norm clipping
  works per parameter tensor, which changes the direction of the update, so it is not used.
- A contrastive loss for labeled pairs and an in-batch ranking loss for catalog groups, each averaged over the
  embedding widths that must stay usable (D4). The stock Matryoshka wrapper in sentence-transformers could not be
  used for this model even in Python: it also truncates token states, and the projection makes the sentence
  embedding (1024) wider than those (384).
- Only catalog-group batches are kept free of repeated texts (a repeat would be a false negative for the ranking
  loss). Labeled pairs repeat texts by nature and the contrastive loss does not care, so they fill whole batches.
- A catalog-group batch of one example is skipped for that pass. Its loss and gradient are exactly zero, but an
  optimizer step on it would still move the weights along leftover momentum and consume a schedule step.
- A width in `--dims` that the model cannot produce is replaced by its native width rather than dropped, so a
  384-wide model is trained at 384 and 256, never at the 256 prefix alone while being served at 384.
- The first warm-up step has a non-zero rate, where the common formulation spends it on a rate of exactly zero.

**Saved models are plain sentence-transformers directories**, written with Hugging Face tensor names, so they come
back through the audited loader (D2) and PyTorch can open them (`just verify-model`). The pooler and pretraining
heads a published checkpoint may carry are not written; nothing in an embedding pipeline reads them. A directory
is never written into if it holds anything, because a model's content hash is its identity (D12).

**Checkpoints are model directories, and a resume is the same run.** A run is checkpointed at the end of every
pass. A checkpoint is what the run would have produced had it stopped there (so it can be calibrated, which is how
the best pass is chosen) plus the optimizer state and a small state file. What carries the weight:

- **Parameter ids.** Burn's optimizer files its state under parameter ids, and ids are made up afresh whenever a
  module is built, including when a checkpoint is loaded. A resume that only loaded the weights and the optimizer
  file would succeed and then never find any of that state, continuing as if it had none. So the state file
  records every id in traversal order and they are put back first. A unit test resumes with and without that
  step: with it the result equals the uninterrupted run, without it the result differs. The entry point that
  loads optimizer state is private to the crate for the same reason: it is only correct after the ids are back.
- **Run identity.** The batch order, the schedule, and what each batch contains are functions of the options,
  the data, and the token limit. The state file records all three (the data as a fingerprint), and a checkpoint
  is refused by any run that differs in one, since it would otherwise continue differently while looking normal.
  The encoder must also have been loaded from the checkpoint directory itself. Options are compared for equality
  after a trip through JSON, so the trainer's `serde_json` is built with `float_roundtrip`; without it about one
  `f64` in ten reads back one bit off and such a checkpoint could never be resumed.
- **Whole or not at all.** A checkpoint and the final model are written to a sibling directory, flushed to disk
  (files and directories), and renamed into place; nothing existing is overwritten. The final model is written
  the moment training ends, before the closing measurements, so the last pass is never held only in memory for
  longer than it must be.
- **Found out early.** The checkpoint directory is created and read before training, must lie outside the output
  directory, and must not already hold a later step. Each of those was otherwise a run lost at its first
  checkpoint.
- **Divergence stops the step.** A loss or a gradient norm that is not finite ends the run before the optimizer
  is applied. Checking only the loss let a step with a finite loss and an infinite gradient through, and the next
  checkpoint would have been written from NaN weights.
- **Pruning is conservative.** `--keep-checkpoints` counts the run's own checkpoints across resumes, recognized
  by their state files; it never deletes another run's, nor the checkpoint the process was loaded from (the model
  re-reads its tokenizer and configuration files from there on every save). Failing to delete one is a warning.

Without dropout a resumed run ends on bit-identical weights (verified on the real model). With dropout it differs
only in the random masks after the resume: the backend's random stream is seeded, not saved.

**Cost.** One step on 32 pairs of 256-token texts takes about 9 s and 9 GB on a laptop CPU and about 1.3 s on its
GPU through Metal. Metal and CPU runs from the same seed without dropout agree to three decimal places in the
resulting similarities. The CUDA feature compiles and has not been run.

**What this does not show.** Everything above establishes that the trainer computes what the reference computes.
Whether fine-tuning improves verdicts is a question for real labeled data and a held-out calibration.

## D14. A tenant, once set, must be named

A record may carry a tenant (the merchant it belongs to). It is stored with the record and filters searches. It
is also checked: once a record has a tenant, an `Evaluate` or `Delete` for its id must name the same one. Naming
another is refused, and so is naming none. The second half matters as much as the first: the mistake the check
exists for is a caller that has lost track of whose record it is editing, and a check that can be skipped by
leaving a field out does not catch that. A record stored without a tenant takes the first one a revision names,
which is how a catalog indexed before tenants were in use gains them.

This is a consistency check and not access control. The server does not authenticate anyone (see the README's
limitations), so a caller that can reach it can always supply the right tenant.

## D15. Logs are NDJSON, and results are output

Both binaries log one JSON object per line and offer no other format. They are run by schedulers and read by log
pipelines far more than by a person at a terminal, and a second format is a second thing to keep correct; the
first version of the trainer had a text mode and a JSON mode, and the two had already drifted apart in what they
reported. A person reads the one format with `jq`. The setup lives in one crate, `similarity-log`, so the two
binaries cannot diverge.

What a line is decides where it goes, and whether it can be filtered:

- **Logs** go to stderr under `RUST_LOG`. Progress and commentary are optional by nature.
- **Results** go to stdout and are never filtered. What a training run measured is its output, in the same sense
  as the calibration report in D11; routed through the log filter, `RUST_LOG=warn` produced a successful run that
  reported nothing.
- **Failures** go to stderr and are never filtered, panics included (a hook replaces Rust's multi-line message).
  The line a reader most needs is the one that says the run failed, so it has to be parseable and present.

Fields are logged as numbers and booleans, rounded where that helps, never as preformatted strings; an optional
value that is absent is left out, not written as `null`. Nothing logged can be `NaN` or infinite, because the
trainer ends a run on the first loss or gradient that is (D13), and JSON has no way to write either.

Two things stay text: the reports `model fetch`, `compare`, and `calibrate` print on stdout for a person, and
usage errors from the argument parser, which happen before anything has started. Burn's own training dashboard
was not used: it comes with `burn-train`'s learner, and a terminal UI is the wrong output for a job run over SSH
or in a pipeline.
