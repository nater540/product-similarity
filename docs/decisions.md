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

**A field's weight may not be less than a millionth of the total.** A field's block in a composite has length
`sqrt(w / total)`, and a block shorter than `1e-6` is read as "this record has no value for the field". A weight
below one part in a million million of the total would therefore make a field vanish from every score, although
each weight on its own is valid. `Schema::new` refuses a weighting a million times more lopsided than that
cut-off allows, which no real configuration is near.

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
  after a trip through JSON, so `serde_json` is built with `float_roundtrip` (workspace-wide, D19); without it about one
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

## D16. Quantities are compared, not embedded

A product record has numeric fields as well as text: a weight, a volume, a THC and a CBD content, each with a
unit. The obvious way to take them into account is to render them as text and embed them as four more fields.
Measured with the shipped model (`mdbr-leaf-mt`, 256 dimensions), scoring each pair of strings as a `name` with
`record-similarity compare`:

| Before | After | What it is | Similarity |
|---|---|---|---|
| `1 g` | `1000 mg` | The same quantity | 0.742 |
| `3.5 g` | `3.5 grams` | The same quantity | 0.970 |
| `weight 3.5 g` | `weight 0.125 oz` | The same quantity (roughly) | 0.832 |
| `THC 21.3 %` | `THC 22.8 %` | A routine batch update | 0.892 |
| `THC 18 % to 22 %` | `THC 19 % to 23 %` | A routine batch update | 0.951 |
| `3.5 g` | `28 g` | Eight times the size | 0.758 |
| `THC 100 mg` | `THC 10 mg` | A tenth of the potency | 0.924 |
| `THC 100 mg` | `THC 1000 mg` | Ten times the potency | 0.961 |
| `THC 22 %` | `THC 22 mg` | A different kind of product | 0.873 |
| `weight 3.5 g, THC 22 %` | `volume 355 ml, THC 10 mg` | A different kind of product | 0.727 |

The ordering is backwards. A tenfold change in potency scores higher than a routine batch update, and an
identical weight written in other units scores lowest of all. Embedded, these fields would add noise to every
verdict. So a quantity is never embedded. It is a `Quantity { min, max, unit }` (a single value is a range whose
ends are equal), compared as a number, which gives three signals the text model cannot:

- **How much it changed**, after unit conversion, as a *factor*: the larger value over the smaller, so it is always
  at least 1 and "eight times bigger" and "an eighth" both read 8. `1 g` to `1000 mg` is 1; `3.5 g` to `28 g` is
  8. For a range it is the factor of whichever end moved more. A move between zero and a value above zero has no
  factor; it is reported as a `zero_crossing` and counts as beyond any limit.
- **That it stopped being comparable.** A potency in `%` becoming a potency in `mg` is not a bigger or smaller
  number. It is reported as `INCOMPARABLE`, with both units.
- **Which measures a record has.** One flag per evaluation, `measure_changed`, is set when a quantity became
  incomparable or when one quantity went away as a different one arrived (a weight for a volume: flower becoming
  a beverage). Filling in a quantity that was simply missing does not set it.

**Units.** A small table in `similarity_core::quantity` converts mass (to mg), volume (to ml), and percentages,
matched without regard to case or spacing. A unit it does not know is never an error: it is its own dimension,
comparable only with the identical unit. The real unit vocabulary is not known here (the data request asks for
it), so the table has to be able to lag behind the data without refusing a record. Extending it is one line.

**Report-only by default.** A quantity is defined by naming it under `[quantities]`, and every response then
carries its comparison. It rejects only where the config asks: `reject_above_factor` and `reject_incomparable` per
quantity, `policy.reject_on_measure_change` overall. `config.example.toml` names four quantities and sets no
limit, because a limit is a measurement in the same sense as a threshold (D3) and there is nothing to measure it
on yet: real product data never enters this repository, and the sample pairs are synthetic. `calibrate` prints,
per quantity, the factors seen in edits beside those seen in recycles, which is what a limit is read from.

**Out of the score.** Quantities never enter the weighted similarity. Folding them in would change what
`policy.threshold` means, invalidate every calibration done so far, and let a wholesale change of text hide behind
four unchanged numbers. Kept apart, the text thresholds are untouched: `just calibrate` printed the same overall
ranges (edits from 0.941, recycles up to 0.850) and the same suggested threshold (0.895) before and after
quantities were added to the sample pairs. `FindSimilar` ignores them too; it compares vectors.

**The anchor rule carries over (D8).** A quantity only one version has cannot be compared, so alone that rule
would let any quantity be swapped in two steps: remove it, then send another. The anchor therefore keeps the first
value of every quantity the record has ever had, exactly as it does for text. A quantity that comes back is
`RESTORED` and compared with that first value, an ordinary accepted write never replaces a value the anchor holds,
and a `FORCE` write resets them along with the anchor vector.

**What counts as a change.** A change in a quantity alone, with no text touched, is a change to the record: it is
written and bumps the revision. The same amount in another unit is not (`1 g` re-sent as `1000 mg`), for the
reason text re-saved with different whitespace is not (D9): re-submitting what is stored must not churn the store.

**The layout does not change (D12).** Quantities are stored in the point's payload, beside the vectors and part of
neither. `Schema::layout_id` leaves them out, so adding or removing a quantity needs no new collection and no
re-index. A point written before quantities existed has no such payload keys and reads back with none; the first
revision that carries a quantity reports it as `ADDED` and the anchor takes it.

**Open questions, and the default in force until real data answers them.** The data request
(`docs/data-request-prompt.md`) asks for each.

| Question | Default |
|---|---|
| Which unit strings actually occur? | The built-in table, matched without regard to case or spacing; anything else is comparable only with itself. |
| Is THC ever given as `mg/g`, or per package instead of per item? | `mg/g` is an unknown unit, its own dimension. Add it to the percentage dimension (factor 0.1) only once it is confirmed to occur. |
| Are `min` and `max` always both set? | A caller that has one sends it as both. |
| Does zero mean "none" or "not measured"? | Zero is a value. A move between zero and anything else is a `zero_crossing`, and rejects only under a factor limit. |
| Is there a pack-size or item-count column? | Not assumed. It would be a fifth `[quantities.<name>]` table and no code. |
| Where do the limits go? | Nowhere yet: the example configuration ships with none. |

**What this does not do.**

- There is no drift check for quantities. `reject_above_factor` applies to each revision against the one before
  it, so a quantity can still be walked a long way in steps that each stay under the limit. The anchor's first
  values are stored, so reporting the factor against them is a small addition if real data shows the need.
- It knows nothing about what a quantity means. A quantity "per item" where the item is a pack and one where the
  item is a piece are simply two numbers; the caller has to be consistent.
- It cannot say where a limit belongs, or whether zero means "none" or "not measured". Those are questions for
  the real data.

**A factor is always a number that can be written down.** Two valid amounts can be so far apart that their
quotient overflows (`1e-300 g` against `1e300 g`). An infinite factor is not a JSON number: a calibration event
carrying it would hold `null`, and the run could not be read back. Such a change gets the largest finite number,
which is beyond every limit, as it should be.

## D17. A calibration is also data: flat events, intervals over owners

`calibrate` printed a report for a person and nothing else, and read only a four-key format that drops everything
a data export knows about a pair (its id, its owner, its split, whether it is a variant change). Real labeled
pairs make three more things necessary: reading the export as it is, saying how sure an error rate is, and
keeping a run in a form another run can be compared with.

**Two input formats, one pair.** The strict format stays what it was, for pairs written by hand: four keys, and a
typo is an error. `--full` reads `pairs.full.jsonl` leniently. Each record is reduced to what the schema defines,
and what that leaves out is **counted by key and reported**. Leniency without the count would turn a misspelled
`descripton` into a slightly weaker score that nobody could trace. A pair labeled anything but `edit` or
`recycle` is scored and shown, and takes part in no count, no suggestion, and no rate: the data request expects
many same-brand variant changes to come back `unsure`, and those are the pairs most worth looking at.

**Every pair has an id.** The export's `pair_id`, or else a hash of the pair's content, so that the same pair in
two runs can be matched up even from a file that never had ids. Two pairs with one id are refused.

**The events are flat.** `--ndjson` writes the calibration through `similarity-log` as result events (D15): one
`pair` line, then one `pair_field`, `pair_quantity`, and `pair_reason` line per item, all carrying the `pair_id`.
A single nested event per pair was the first design and cannot be written: `tracing` records scalar values under
names fixed at the call site, so a map keyed by field name, or a list of reasons, could only be a preformatted
string, which D15 forbids for good reason. Flat lines are also what `jq` and a table loader want. Each reason has
a stable code (`Reason::code`), so nothing has to parse the sentence. Similarities are written to six decimal
places, below what a 32-bit cosine resolves, so that judging the lines again under another configuration gives
what scoring afresh would.

**One source for every number.** The text report and the events are both rendered from `offline::stats`. Two
renderers that each computed their own ranges would drift apart, as the trainer's two log formats once did (D15).

**Intervals resample owners, not pairs.** The two error rates and the operating points carry a 95% interval from
1,000 seeded bootstrap resamples. What is resampled is the owner, with all of its pairs: edits by one brand are
written by the same people about the same product line and are not independent, so resampling pairs would give an
interval that is too narrow, and narrower the more one brand dominates. That needs every labeled pair to name an
owner and at least two owners; otherwise no interval is printed, which is honest for a file that cannot support
one. A rate with no errors has a zero-width interval. That is a limit of the method on small data and the README
says so; it is not evidence of a zero rate.

**Operating points** answer the question a threshold poses: to let through at most 1%, 2%, or 5% of recycles, how
many edits are rejected? They use the overall score alone. A target that allows no more recycles than the one
before it is left out, since it would repeat it.

**A quantity limit is suggested from the edits.** The first reading of the quantity lines was "the limit belongs
between the two ranges". That is the logic of a threshold and it is wrong here: most recycles do not change a
quantity at all (another strain at the same weight), so the smallest factor among recycles is always about 1 and
there is never a gap. A quantity limit is an *additional* way to reject. What it must not do is reject an edit,
so the suggestion is the geometric midpoint (factors are ratios) between the largest factor any edit shows and
the next factor above it among recycles, with the number of recycles it would catch. A quantity no edit changes
gets no suggestion: there is nothing to place a limit above.

The generator behind the bootstrap is the trainer's `SplitMix64`, moved to `similarity-core` so that the server
can use it without depending on the trainer (D13). Its sequence is pinned by a test, since a trained model and a
reported interval are both reproducible only while it does not change.

**Comparing two calibrations judges again; it does not embed again.** `calibrate-diff` reads two event files and
reports every verdict that changed. For weights, floors, thresholds, and quantity limits it needs only one:
`--config-b` re-applies them to the per-field similarities and per-quantity factors of the first run
(`Evaluation::rejudge`), with no model loaded. An embedding cache was considered for this and dropped: the only
changes it would serve are a new model or a new set of fields, and those are a new calibration by definition.
The events therefore carry the schema they were scored under (`schema_field`, `schema_quantity`), and judging
again under other field or quantity names is refused. On the bundled sample a real second run and a `--config-b`
judgment report the same flips.

**The API carries the same codes.** `EvaluateResponse.reason_details` is each reason as data: a `ReasonCode`, and
where they apply the `subject`, `value`, and `limit` that `Reason` already had. The code is an enum, like every
other categorical on the wire, and each value is named `REASON_CODE_` plus the `Reason::code` string in capitals;
a test holds the two together, so a reason read from a response and one read from a calibration go by one name.
The engine now returns the `Reason` values and `grpc.rs` renders both the sentence and the detail from that one
list, which is what keeps them in step. The sentences stay, not deprecated: they are what an older client shows
and what a newer one falls back to for a code it does not know. Units are not repeated on a reason, because its
`subject` names the `QuantityScore` that has them, and a property holds that join for every generated history.
Two things the suggestion also asked for were left out. A word-level diff of a modified field cannot be made here:
the server keeps a hash of each field's text, never the text, and the caller already has both versions. A
neighbor search on rejection would show a neighbor only above some similarity, and that number has not been
measured; a caller can ask `FindSimilar` today.

## D18. A model directory says where it came from

A model's fingerprint says which model produced a vector. It does not say how the model came to be. After two
rounds of fine-tuning, nothing recorded which base, data, and options made a directory under `models/`, and
thresholds belong to one specific model (D3). That cannot be reconstructed later, so it is written when the model
is: every directory `model fetch` or the trainer writes holds a `provenance.json`.

- **Fetched:** the repository, the revision, and each file with its SHA-256 and whether that was checked against
  a pinned checksum.
- **Trained:** the base model's directory and digest, with the base's own provenance nested inside (so the chain
  back to a published model is in one file); each data file with the SHA-256 of its text and the number of
  examples it gave; the existing fingerprint of the examples; the options; the token limit; the device; and the
  checkpoints the run was resumed from. A resumed run keeps the base it first started from. A checkpoint carries
  the same record with the step it holds.
- **Both:** the tool, its version, the commit it was built from, and whether that tree was dirty.

**It records inputs, not results.** The first design also put the final `separation` measurements in the file.
The trainer saves the model the moment training ends, before measuring anything, so that the last pass is not
held only in memory through minutes of scoring (D13); and a model directory is written whole and never modified
in place. Adding results afterwards would break the second rule to serve a convenience. What a model scored is in
the run's result stream.

**It is not part of the model.** `ModelFiles::digest` hashes the files the encoder is built from and the
tokenizer, and nothing else, so the file changes no fingerprint and invalidates no stored vector (D12). That
digest used to be computed inside `Embedder::load`; it is now a function of its own, so that the trainer can
compute the base's digest and the digest of what it just saved without building an embedder. The pinned model's
fingerprint is unchanged by the move (`f46e48ee0ea2dd02`). `SentenceEncoder::save` copies only what it read plus a
fixed list of side files, so a base's provenance is never carried into a model trained from it as if it were that
model's own; it is nested under `base` on purpose instead.

**Nothing depends on it.** A model without the file trains, loads, and serves as before. The server logs what the
file says on its `model loaded` line, warns if the digest in the file is not the digest of the files beside it
(someone replaced the weights and left the note), and never refuses to start over it.

**The commit comes from a build script**, shared by the two binary crates. It reads `RECORD_SIMILARITY_COMMIT`
if that is set (the Docker build has no `.git`; `just image` passes it), otherwise asks `git`, otherwise says
`unknown`. It runs on every build, because whether the tree is dirty can change without any file of the crate
changing; the crate is only rebuilt when the answer does. It lives in the binary crates and not in
`similarity-core` so that a change of answer relinks two binaries instead of rebuilding the workspace.

## D19. The rules a verdict rests on are stated as properties

D7 (a rejected revision writes nothing) and D8 (the anchor never loses a value) are the product's safety
properties, and they were covered by scenarios someone had thought of. Scenarios do not cover the request nobody
thought of, and the engine's write path was about to be changed again (quantities, a bulk import). So the rules
are also written down as properties and checked on generated inputs, with `proptest` as a dev-dependency (MIT or
Apache-2.0, never in a release build, built without its process-forking features).

**What is checked.** In `similarity-core`: the composite identity of D5; that a comparison names and scores every
field from which sides have it; that extending the anchor keeps what it holds; that judging again equals
evaluating afresh (D17); that a quantity's factor is the same in both directions, at least 1, and unmoved by
rewriting both sides in another unit (D16); that judging quantities never touches the text verdict; and that a
text embeds the same alone and in any batch (D9). In `similarity-server`: that the suggested threshold is what
trying every cut finds, and a **state machine**: random sequences of `Evaluate` (every mode, valid and malformed,
with and without tenants and quantities), `Delete`, and `FindSimilar`, with every rule of D7, D8, D14, and D16
checked after every request against what the store held before it.

**It can fail.** Three deliberate breaks of the engine (writing a rejected revision, replacing the anchor's
quantities, counting a write twice) were each caught within the default 64 cases and shrunk to a two-request
sequence. A property that cannot fail is decoration.

**What it found.** One real thing: a number read from JSON was not always the number written. `serde_json` is off
by one bit for about one value in ten unless built with `float_roundtrip`. The trainer already enabled that for
its checkpoints (D13), and because cargo unifies features across a build, the server crates had it under
`cargo test --workspace` and not under `cargo test -p similarity-core`: the same file parsed to different numbers
depending on what else was being compiled. Nothing visible depended on the last bit (quantities are compared
within a tolerance), but a number that depends on the build is a bug waiting for its symptom. The feature is now
set once, for the workspace.

It also found a mistake in a test, which is worth recording for what it shows: `Schema::compose` requires unit
vectors, says so, and does not check. Every caller gives it unit vectors (the encoder normalizes its output, and
reused blocks are normalized again). The composite identity does not hold otherwise.

**Cost.** Each property runs 64 cases by default, and the whole of it adds well under a second to `just test`.
`PROPTEST_CASES` raises the count for a thorough run; 5,000 cases of everything take a few seconds.

## D20. A catalog is imported by a command that only creates

Loading a catalog was one `Evaluate` call per record, and so was every re-index, which D12 makes necessary after
any change of model. Worse, a re-index reset every record's drift history: the anchor became whatever the record
said on the day of the re-index, because the service stores vectors and never the text, and so could not rebuild
the original anchor under a new model.

`record-similarity import` reads an export (one record per line) and writes through the store, a batch per call
to the model. Three choices in it:

**It only creates.** `Engine::import` writes an id that is not stored, at revision 1, and leaves an id that is
stored exactly as it is, down to its timestamp. An import that could update would be a way to replace a baseline
and an anchor without a verdict, which is everything D7 and D8 exist to prevent; and "which records did the
interrupted run get to?" would need an answer. Creating only, it is safe to run twice and is resumed by running
it again, with no bookkeeping. The same request rule holds as anywhere else: an id stored under another tenant
is an error for that record, not a skip (D14).

**The original version, where the caller still has it, becomes the anchor.** A line may carry `anchor`: the
record as it was first created. Its text is embedded under the current model and its quantities are kept, and
anything it lacks takes the record's current value, which is exactly what `extend_anchor` and `first_values`
would have made of the anchor over the record's life. Drift is then measured from where the record really began,
and a field or quantity that was removed before the import is still guarded against being refilled with
something else. Only the caller's database can supply this; the service cannot.

**A bad record fails alone.** It is reported with its line, id, and reason, the rest is imported, and the command
exits non-zero at the end. `--check` runs the schema's validation over the whole export with no model and no
store, since the usual problems (a misnamed field, a quantity sent as text) are all visible there.

Everything is embedded before anything is written, and nothing is written if the model fails. The records of a
batch are embedded in one call, on the calling thread: a command that is the model's only user has no use for the
server's queue (D9). On a laptop CPU 2,000 four-field records import in 19 seconds.

Not built: a streaming `Import` RPC. The command covers the need as long as it can run next to Qdrant, and an RPC
that writes without judging is a larger thing to expose than a command an operator runs.

## D21. A long field is tokenized a part at a time, until the token limit is full

A tokenizer walks the whole of a text before it truncates to `max_tokens`. Only the first 256 tokens of a field
are ever used, but the cost of a field grew with its full length: scoring one record whose description was 1 MB
of text took 0.90 s where an ordinary one takes 0.22 s, and 3 MB took 2.2 s (measured with `compare`). There is
one embedding thread (D9), and a gRPC message may be 4 MB, so a handful of such requests held up every other
caller.

`TextTokenizer::encode` now hands the tokenizer a head of a long text: at least `max_tokens * 64` bytes (16 KB at
the default limit), ending just before the next space, tab, or line break. If that fills the token limit, it is
the answer, since whatever follows could only have been truncated away. If it does not, a head four times as long
is tried, and so on up to the whole text. Comparing two records that each carry a 1 MB description now takes
0.33 s, and with 3 MB each 0.34 s, where two ordinary records take 0.24 s.

**Why this and not a size limit on requests.** Refusing a long field would make requests fail that succeed
today, for a caller who has done nothing wrong but store a long description. Reading less of the text changes
nothing a caller can see, so long as the tokens are the same.

**Why the tokens are the same, for every text.** A head that stops at a space, a tab, or a line break stops
between two words for every tokenizer, so its tokens are exactly the first tokens of the whole text. The head is
accepted only when the tokenizer itself reports the limit full. Nothing is assumed about how many bytes a token
takes. A property test (`a_text_is_tokenized_as_if_all_of_it_had_been_read`) compares the two on generated text,
and the parity tests against PyTorch pass unchanged.

**The first version assumed, and was wrong.** It cut every text at `max_tokens * 64` bytes, on the reasoning that
a token is a few bytes and 256 of them lie well inside 16 KB. That holds for ordinary text and not for text built
to break it: 16 KB of zero-width spaces or combining accents come to no tokens at all, and one unbroken
20,000-letter word comes to one. Put in front of every field, each pushed the real text out of what was read, so
two unrelated products embedded identically, scored 1.000, and were accepted, where without the padding they score
0.519 and are rejected. The first two paddings cannot be seen on a page. It was found while planning the fuzz
targets (`docs/plans/14-fuzzing.md`) and reproduced with `compare` on the real model; the same three inputs now
score 0.519, 0.519, and 0.583. `padding_cannot_push_a_text_out_of_what_is_read` and
`padding_a_text_does_not_hide_it_from_the_encoder` (real model) keep them as tests. The lesson is the one D19
records: a claim that holds "for any text" needs a test that generates text.

**What it does not bound.** A text with no space, tab, or line break anywhere is read whole, as before: two
records with 1 MB of unbroken letters each take 1.51 s. A text whose heads never fill the limit is read in parts
that add up to a third more than the text. So this makes long ordinary fields cheap; it is not a defense against
a caller who wants to be slow. That is the job of a limit on message size.

**What is not cut.** The text's fingerprint is still taken over the whole of it, so an edit beyond the tokens that
are kept is still an edit: the field is reported as modified, at a similarity of 1, exactly as an edit beyond the
256th token always was. Whitespace normalization and hashing still read the whole field; they are linear and
cheap.

The logic is in the tokenizer wrapper that serving and training share, so a model is still fine-tuned on exactly
the token sequences it is served with.

## D22. Verdicts on hard cases are pinned, and checked without the model where they can be

"Run `just test-real` after touching the encoder" was a rule, and a rule is only as good as the memory of whoever
is making the change. The cases a change is most likely to break are also the ones nobody thinks to try by hand:
a weight rewritten in ounces, a field blanked in one edit and refilled in the next, a Cyrillic letter in a name.

`data/regression.jsonl` holds 25 such pairs, synthetic, in the full pair format.
`data/regression.snapshot.ndjson` is what the pinned model and `config.example.toml` make of each one.

**The snapshot is a calibration run, not a new format.** It is the output of `calibrate --ndjson` (D17) with the
`timestamp` taken off each line, so `events::read` reads it and `calibrate-diff` compares it. The timestamps go
because they would make every regeneration a whole-file change, and the point of committing the file is that its
diff shows what moved.

**Two checks, because two kinds of change move a verdict.** A weight, a floor, a threshold, a quantity limit, or
the judging code changes what is made of the similarities, not the similarities. `tests/regression_snapshot.rs`
therefore judges the committed snapshot again under the current `config.example.toml` and scoring code
(`diff::rejudge`) and requires that nothing comes out different. It needs no model and runs in `cargo test` on
every machine. Only a change to the similarities themselves (the model files, the encoder, the loader, the
tokenizer, pooling, truncation) needs the model: `tests/real_snapshot.rs` scores the cases afresh, and CI runs it
with the model cached under the hash of `fetch.rs`.

**The check is by tolerance, never by bytes.** `calibrate-diff` learned to report what moved without a verdict
flipping: a similarity by more than a tolerance, a kind of change, a factor, a reason. The tolerance is 0.001.
The parity tests hold an embedding to a cosine of 0.99999 with PyTorch's, so the same model on another machine
should agree to about five decimals, and 0.001 is well clear of that and well short of a real change. That
reasoning has not yet met a Linux runner: the snapshot was made on macOS, and the repository has no remote for CI
to run on. The first CI run is where the figure is confirmed or raised.

**A case on a border is refused.** `just snapshot` will not write while any case sits within the tolerance of the
threshold, floor, or limit it is judged against (`diff::borderline`). Such a case would flip on the noise between
two machines, and a gate that fails for no reason gets switched off. The case is changed instead.

**A pair may carry an anchor.** The blank-and-refill swap is two revisions and an anchor (D8), and a before/after
pair scored its second step as a field being added, which is left out of the score: the opposite of what the
server does. A pair may now say what the record was first created as. It is built as an import with an original
version is built (D20), and a property scores generated anchored pairs through the offline scorer and through an
engine import followed by an evaluate, on the same weights, and requires the same measurements, reasons, and
verdict.

**It pins behavior, not correctness, and it is not a calibration.** Four cases are same-brand variant changes
labeled `unsure`: three are accepted and one rejected, and the snapshot holds that without claiming it is right.
One case, a weight that was dropped and comes back eight times as large, is a recycle the shipped configuration
accepts, because it sets no quantity limit (D16). The cases were picked to be hard, not sampled, so no threshold
or limit may be read from them (D3).

**The README's quoted reports are checked too.** Each calibration report quoted under "Choosing thresholds" is
preceded by a `<!-- measured: calibrate ... -->` comment, and a test in the same file runs that calibration and
requires every quoted line to be one it prints.

Regenerating is local (`just snapshot`) and CI only checks: a job that rewrote a pinned file would be an
unreviewed change to it.

## D23. A run leaves a directory behind

What a fine-tuning run measured was whatever someone remembered to redirect: results to one file, logs to another,
and a calibration's data only if `--ndjson` had been typed. The first real training runs are about to happen, and
a result that was not kept cannot be recovered.

`calibrate` and the trainer now each record a run: a directory under `runs/` with a manifest (`run.json`), every
event the run emitted (`events.ndjson`), and for a calibration read as text, the report. It is on unless
`--no-run` is given. Nothing leaves the machine, and nothing depends on Weights & Biases or any other service: an
exporter to one could read these files afterwards.

**The record is a third layer of the one logging setup, not a second one** (D15). `similarity-log` builds every
layer with the same function, so a line in the record is the line the streams carry. It has a level of its own:
results and failures always, and logs at `info` whatever `RUST_LOG` says. The loss curve is made of `info` logs,
and a copy of the two streams would have lost it whenever someone quietened stderr.

**A calibration is data in either mode.** Read as text, a calibration used to write no events at all. A recorded
one writes them to the record alone while stdout keeps the report, which is the one case where results do not
go to stdout: the same results are in the run, and stdout has what the person asked for.

**`killed` is never written.** A killed process writes nothing, and the trainer installs no signal handler. The
manifest says `running` until the run ends and is then rewritten as `finished` or `failed` (a destructor covers a
panic). The process holds a lock on a file in its run directory, which the operating system releases however the
process ends, so a reader concludes `killed` from a `running` manifest whose lock nobody holds. Readers never
write.

**A resumed run is a run of its own.** Appending to the original run's directory was the suggestion, and is
wrong three ways: the resume may be on another machine, may start from an earlier checkpoint than the last (its
steps would then be in one file twice), and may be started twice by mistake. Each invocation gets a directory,
and a resumed one names the checkpoint it went on from and the run that wrote it. The link is carried by
`provenance.json`, which gains the id of the run that wrote the directory: that is a fact about what produced a
model, so it belongs there (D18), and it is no part of a run's identity for resuming, because a resume is always
another run and must be accepted.

**Everything that could stop a run from being recorded is found out before any work**, as with checkpoints (D13).
After that, a note that cannot be written is a warning: the training matters more than the note about it.

**The id is the time, the kind, and a name**, `20261111T201826Z-train-mdbr-leaf-mt-ft`, so that a directory
listing is in order and needs no index. The name comes from a path the caller chose, so a property holds that an
id is always a single safe path component.

**What it does not do.** It records and lists. Comparing runs, drawing them, and deciding on them are features
4 and 5, which read these files. `serve` and `import` are not recorded: a server is not a run, and an import's
command line may one day carry a secret, which a manifest must not keep.

The crate is `similarity-runs`. It depends on `similarity-core` for the two types a manifest shares with a
model's provenance, the build stamp and a data file; both binaries link `similarity-core` already.

## D24. A model is chosen on a validation split, by a score that needs no threshold

The README told the reader to pick thresholds, and to tune the learning rate, by looking at the holdout. Anything
chosen that way has been fitted to the holdout, and the error rates it reports afterwards are better than
production will be. Two planners found this independently. The export has two splits, and one of them cannot
both choose and judge.

**A third split, carved from the first.** `pairs carve` moves a share of one split's owners (by default from
`calibrate` to `validation`). Whole owners move, because pairs of one brand are not independent. Which owners is
decided by a hash of the owner's name and a salt, with no randomness and no state: the same command moves the
same owners, carving twice moves nothing more, and a larger share keeps every owner a smaller one moved. The
share has no default. It is not a threshold, but it is a judgment about how many labeled pairs a comparison
needs, and that depends on the data.

**Both binaries select by `split` alone.** The trainer trains on `calibrate`, compares on `validation`, and
reports on `holdout`; `calibrate --split` reads any of them. The alternative was a fold computed in memory from
an owner hash in each binary (which would allow k folds). One carved split was chosen because the two binaries
then cannot disagree about which pairs are which, and a file is something a person can look at.

**Catalog groups of measured owners are left out of training.** A catalog group has no split of its own. One
whose owner is in the validation or holdout split is skipped, or the brand held back for measuring would be
learned from by another route. A group that names no owner is kept and counted.

**The trainer compares by AUC, not by errors at a threshold.** The area under the curve is how often an edit
scores above a recycle: it depends only on order. "Errors at the configured threshold" was the obvious measure
and is wrong for this, because the threshold was measured on the base model and does not transfer (D3): a
fine-tuned model that moved every score would be judged on a scale that is no longer its own. The function is in
`similarity-core`, pure, with a property that checks it against counting every pairing. `calibrate` reports the
same number over the overall similarity, so the two can be compared.

**Every candidate is a model on disk, scored as it would be served.** The model the run started from, each
checkpoint (loaded back from what was just written), and the final model. The base is a candidate because
training can make a model worse, and on the bundled sample, which the base already separates completely, it is
the one chosen. A tie goes to the earlier step. `--keep-checkpoints` keeps the newest few and the best.
`--out` is still the final model, saved the moment training ends (D13); the best is named, not copied.

**The holdout is consulted once.** With validation pairs, the only held-out line the trainer prints after
training is for the model that was chosen (`stage: "selected"`, or the usual `after` line when the final model is
the one). The `before` line stays: the base model is not a choice.

**What was left out.** A resumed run scores the checkpoint it resumed from, its later checkpoints, and its final
model. It does not score again the checkpoints an earlier process wrote, so its choice is among what this
process saw. Results are never written into a checkpoint directory (D18); they are in the run's events (D23).
Whether a validation split of a given size ranks checkpoints reliably can only be seen on real data.

## D25. The tests are tested: mutation testing on the files whose boundaries are verdicts

A test suite says what the code does on the inputs someone thought of. It does not say which of the code's
comparisons could be changed without any test noticing. For most code that is a curiosity. Here a comparison is
a verdict: `<` against `<=` at a threshold decides whether a record is rejected.

`just mutants` runs cargo-mutants over eight files: `score.rs`, `quantity.rs`, `schema.rs`, `vector.rs`, and
`record.rs` in `similarity-core`, and `engine.rs`, `offline/stats.rs`, and `offline/calibrate.rs` in
`similarity-server`. Each crate's mutants are tested by that crate's own library tests, which is stricter than
letting the engine's scenarios vouch for the core's arithmetic, and much faster.

**What the first runs found.** Eighteen survivors in 392 mutants, each a real gap:

- A score exactly at the overall threshold, and at the anchor threshold, was held by no test.
- Nothing tested that two requests for one id are judged one after the other. The lock was there; a test with a
  store that yields on read now fails without it, because both requests then create the record.
- `field_range`, which a field's floor is read from, had only been tested on input where it returns nothing.
- The threshold suggestion used the midpoint of two neighboring scores. Between two scores one float apart the
  midpoint rounds to the lower one, and two scores that a threshold does separate were reported as an error. The
  candidate is now the upper score in that case.
- A bootstrap interval was never checked to lie between 0 and 1.

After a test for each, 400 mutants: 333 caught, 67 that do not compile, none surviving, in 14 minutes at three
jobs on a 12-core machine.

The set grows with the code. With the drift statistics, the histogram, and the engine's observer in it, a run
at `7b14697` tested 517 mutants in 18 minutes: 434 caught, 83 that do not compile, none surviving. The run
before it had left three alive, in the histogram's range checks and in what counts as a recycle caught by drift
alone, and each got a test.

**Why an allowlist of files.** The alternative, excluding what should not be mutated, grows a list of everything
whose only tests need the model or Qdrant, and a new file is mutated by default whether or not its tests can run.
The eight files are where a mutant that survives means a wrong verdict.

**Why one fixed proptest seed.** The property tests draw a new seed each run. A mutant that one seed catches and
another misses would make the survivors differ from run to run, and a gate that changes its mind is ignored. The
recipe fixes the seed and turns off the regression files, which would otherwise be written into a copy that is
thrown away.

**Why it is not in `just ci`.** A quarter of an hour of twelve cores. It is run by hand after a change to what
it covers, as the thorough property pass is. The repository has no remote, so there is no scheduled job to give
it to.

**Excluding a mutant is for one case only:** the change cannot alter behavior. One is excluded, with its reason:
`>` against `>=` at a block length that no block can have, since `Schema::new` now refuses the weighting that
could produce one. Where a survivor showed code that was redundant (a filter that the next comparison repeats),
the code was simplified instead, which removes the mutant with it.

**The tool copies the tree for each job as the job starts.** The first run was made while another change was
being edited, two of its three copies did not compile, and it reported 206 mutants as failing to build. That
reads like a result. A run is made from a clean checkout.

**The trainer's checkpoint rules are a second set** (`just mutants-train`, `checkpoint.rs`), kept apart because
its tests take real optimizer steps and the trainer changes rarely. Its tests are in the library and in the
binary, so both are run. Ten of its 59 mutants survived at first. Eight were gaps: nothing tested that a
checkpoint which says it is at or past the end of its run, or which records the wrong number of parameters, is
refused; that a directory is written under parents that do not exist yet; or that a link to a checkpoint
directory is that directory. Each has a test now. The other two replace the functions that flush a file and a
directory to disk, which changes what survives a power cut and nothing a test can observe; they are excluded,
with that reason. After that: 58 mutants, 39 caught, 19 that do not compile, none surviving, in four minutes.

**What it cannot see.** cargo-mutants swaps operators and replaces function bodies. It never changes a constant
or a method call, so the tolerance in `quantity.rs`, every `saturating_*`, and `min`, `max`, and `clamp` are
beyond it. Those stay with the named tests and the properties (D19).

## D26. Throughput is measured through the API, with the worker saying why

The README could say what a forward pass costs and not what a caller waits. An earlier draft derived a figure for
concurrent requests from the forward-pass numbers, and it was removed for being derived. The gap is real:
between the model and the caller are the queue in front of the one embedding thread, the merging of waiting
requests into one pass (D9), gRPC, the per-record lock, and the store.

**`record-similarity load` drives the real API.** It starts a server in its own process on a store in memory, or
is pointed at a running one, and sends synthetic records through gRPC in three phases: create (every field to the
model), edit (one field), and resave (none). It is a subcommand and not an example so that it is in the image and
can be run where there is no Rust toolchain, and so that its logic has ordinary tests.

**The worker counts what it does.** A client sees how long a request took and nothing about why. The embedding
thread now keeps six counters (calls into the model, requests, texts, refusals, time in the model, time requests
waited), read as snapshots and subtracted for a phase. They are a few atomic additions beside a forward pass of
milliseconds. Nothing is logged per batch: under load that is the volume D15 warns about.

**What it measured** (12 cores, default model): from one caller to 64, edits per second rise from 177 to 470
while the median latency rises from 6 ms to 116 ms, with the model busy throughout and the number of texts per
forward pass rising from 1 to 23. Creating whole records levels off at about 80 a second once a pass is full.
The full table is in the README.

**The loop is closed, and says so.** Each caller waits for its answer before sending again. That measures latency
at a fixed number of callers. Latency at a fixed arrival rate, where a server that falls behind makes its queue
grow, is a different measurement and needs an open loop; the README warns against reading one as the other.

**The store defaults to memory, and Qdrant is never the default.** A load test that wrote into the configured
collection because nobody said otherwise would be a bad afternoon. Against a running server the records go under
a tenant of their own and are deleted afterwards.

**Request lines carry `elapsed_ms`,** and the `evaluate` line's similarities are rounded to six places as every
other event's are (the last line that printed a 32-bit value's conversion noise).

**Left out:** an open-loop mode, a Qdrant store in process, records read from a real export, and a regression
gate on these figures. A gate on wall time would fail on a busy machine and be switched off.

## D27. The drift threshold is read from whole histories, measured first and judged afterwards

`anchor_threshold` shipped at 0.85 and enforced, and nothing recorded where the number came from. D8 explains
the anchor, not the value. A threshold is a measurement (D3), and this one cannot be measured on before/after
pairs: it is about how far a product moves over its whole life.

`calibrate-drift` replays edit histories (`sequences.jsonl`, every version of a product in order) through the
engine and reports, for each, the lowest similarity to its anchor.

**Measured first, judged afterwards.** The obvious replay, under the configured policy, is wrong. After the
first rejected version the stored record no longer moves, so every later version is compared with a baseline the
real product left behind, and the rest of the history is fiction. The replay runs on an engine with the same
fields and weights and every limit cleared, so that every version is written and the anchor follows the engine's
own rules (D8, D20). Each step's measurements are then judged under the configured schema and policy with
`Evaluation::rejudge`. A test replays histories both ways and requires the same first rejection, with the same
reasons, as the server gives.

**It is not placed like the overall threshold.** There is no gap between two labels to put it in: most recycles
are one large step, which the per-step rules catch, and such a history says nothing about drift. What a drift
threshold must do is stay below every legitimate history. What it can add is the recycles the per-step rules
miss. The suggestion is halfway between the lowest legitimate history and the highest missed recycle below it,
and there is none when no missed recycle is below every legitimate history.

**A history has a legitimate part and a recycled part.** `recycled_at` names the first version that is no longer
the product. Without it, a recycled history could contribute nothing to "how far does a legitimate product
drift", and its first rejection could not be called early, on time, or late.

**A truncated history places nothing.** One whose first version is not its creation has already drifted by an
unknown amount. `original`, when the export has it, seeds the anchor through the same path as an import (D20);
without it the history is marked `truncated`, replayed, and left out of the ranges and the suggestion.

**Quantities: counted, not built.** The open question from D16 is whether quantities need a drift check, since
a factor limit holds each version only to the one before it. The run counts the recycles that end further from
a quantity's first value than any legitimate history does, in steps no larger than a legitimate history's
largest (`walked`). A check would be built on that count being more than a curiosity on real data, and not
before.

**The shipped 0.85 stays, and now says what it is.** Removing it would stop enforcing drift for anyone who
copied the example config, and replacing it with the sample's suggestion would be reading a threshold from
synthetic data. Its comment says it is a starting point and where the real value comes from.

**One store, in memory, per run.** The replay needs the engine's write path and nothing durable. `Engine::store`
is readable inside the crate so that the anchor a step is held to can be read before the step.

The drift events are new names (`drift_run`, `history`, `step`, ...), so nothing that reads a calibration is
affected. They carry each field's similarity to the anchor, which is what would let a drift run be judged again
under other weights. That re-judging is not built: the overall anchor similarity is weight-dependent, and
`rejudge` reuses it as measured.

## D28. Runs are compared as text and reported as one file

A recorded run (D23) is only worth keeping if two of them can be put next to each other. Two questions come up
after every pair of training runs: what was different about them, and what did it change.

**`runs compare` answers both in a terminal.** Two training runs are laid out as what went in (the base model's
digest, each data file's hash, every option) and what came out (the losses, the chosen model, the separation
before and after), with every row that differs marked. The one option that was changed is then the first thing
seen. Two calibrations are handed to the comparison that already exists (`calibrate-diff`, D17): a second
comparison of calibrations would be a second thing to keep right. It has no `--fail-on`; the gate is
`calibrate-diff`.

**`runs report` is one HTML file with no script and nothing fetched.** It opens from disk, attaches to a
message, and prints. Charts are inline SVG. A value is read from a native tooltip, and every chart is followed by
a table of exactly what it draws, so nothing depends on color or on a pointer. Lines are told apart by dash
pattern as well as by color, and the six colors were checked for color-vision deficiency in both the light and
the dark theme.

**It lives in the server crate, not in one of its own.** Comparing two calibrations needs the calibration reader
and the comparison, which are there, and which the trainer must never link (D13). The commands are under
`record-similarity runs`, beside `runs list`.

**The page computes nothing.** Training numbers come from `runs::training`, which reads a run's events back by
message name; calibration numbers come from `offline::stats`, which gained the histogram for this. A renderer
that worked something out for itself would be a second place for a number to be wrong.

**A resumed run is one curve.** Its directories are followed back through `continues`, and the processes are
stitched: a process that went on from step 20 replaces whatever its predecessor logged past step 20, since those
steps were taken again. A link to a directory that is gone is an error naming it, not a shorter curve.

**What a chart may show.** One measure per chart (loss, gradient norm, and learning rate are three charts, never
one with two axes), at most six runs, and histogram bins that are for looking at: the page says no threshold is
to be read from where a bin ends, and the only thresholds drawn are the configured one and the one `stats`
suggested.

**Left out:** grouping calibrations under the training run that produced their model, per-field range bars, and
a shared x range across calibrations. Each is a layout refinement over data the page already has.

## D29. A policy is swept on measurements already made; training is swept by hand

Choosing weights, floors, and limits meant editing a config and running `calibrate-diff --config-b` once per
idea. Each of those is free: the settings change what is made of a calibration's measurements, not the
measurements (D17). So a grid of them is judged in one command, `calibrate-sweep`, on one recorded run, with no
model.

**Each point is judged at its own best threshold.** Changing a weight moves every overall similarity, so the
threshold that was right before is not right after, and comparing points at one fixed threshold would compare
how well each happens to suit that number. The threshold search is exact and cheap (every cut between two
neighboring scores is tried), so it is done per point and `policy.threshold` is refused as an axis.
`anchor_threshold` is refused too: pairs have no history (D27).

**The ranking is errors, then margin, then the order written.** Errors under the point's whole policy, floors
and limits included, since fitting those is the purpose. The margin breaks ties because a point that separates
the labels with more room is the better bet on other data. The order of the grid breaks what is left, so that
the ranking depends on nothing but the grid.

**A sweep overfits, and says so.** A grid of a thousand points fitted to a few hundred pairs will find one that
looks better by chance. The report says the counts are optimistic, and `--evaluate` judges the best points on a
second calibration at the threshold chosen on the first, never chosen again: the only numbers in it that were
not fitted. It refuses a second run from another model, since a threshold does not carry between models (D3).
The same rule as for models (D24): rank on the validation split, evaluate once on the holdout.

**Points in the same proportions are one point.** Only the ratios of weights matter to a weighted mean, so
`2:4` and `1:2` are the same configuration. They are collapsed, and a point's id is a hash of the configuration
it amounts to, so that the same point has the same id in every sweep.

**The winner is printed as something to try.** A sweep is a search, and what it finds is a hypothesis. The
report prints the settings and says to run `calibrate` under them and compare with `calibrate-diff`.

**The training sweep is not built.** The suggestion asked for one command that trains at every point of a grid
of training options and seeds, and scores each. What was built instead is everything such a command would rest
on: every option that changes a training step is a flag (`--warmup-ratio`, `--margin`, and `--scale` were not),
every run is recorded with its options (D23), a model is chosen on a validation split by a score that needs no
threshold (D24), and runs are compared and drawn (D28). A loop that calls the trainer is a few lines of shell on
top of that. An orchestrator of its own is worth building once real fine-tuning has shown which options are worth
sweeping, and how many runs a sweep can afford: each point is a whole training run. Two things in the suggestion
would have been wrong in any case: sweeping the number of passes by picking checkpoints (the learning-rate
schedule depends on the total number of steps, so the checkpoint after pass two of four is not the model two
passes would give), and judging trained models at the configured thresholds, which belong to the base model.

## D30. A candidate policy runs in the shadow of the live one; a candidate model does not

A changed policy could be tried on labeled pairs (`calibrate-diff --config-b`, `calibrate-sweep`) and then only
by deploying it. Pairs say whether it separates the labels. They do not say how many of the requests the service
really gets it would decide differently, or which. A `[shadow]` table answers that: every judged revision is
judged again under the candidate, and the disagreements are written down.

**The engine tells an observer; it never asks one.** `Engine::with_observer` takes one `Observer`, whose
`observe` returns nothing. It is called once for each successful `evaluate` of a record with an id, after
anything the request writes is written (a write that failed returned its error and is not observed), with the
record's lock still held, so what is observed about one id is in the order it was judged. The observation
borrows what the engine already has in hand (the baseline, the candidate's vectors and quantities, the live
evaluation), so nothing is copied for it. That shape is what makes "a shadow cannot change a verdict" true by
construction instead of by care: there is no value for it to return and nothing mutable for it to reach. A
property holds it anyway (`a_shadow_never_changes_an_answer_or_a_write`): any sequence of requests gets the same
answers and leaves the same store with a candidate watching, including one that fails on every revision.

**The candidate judges as the engine does.** `score::evaluate` and `judge_quantities`, under the candidate's
schema and policy, on the live vectors, against the live baseline and anchor. Not `Evaluation::rejudge`, which
keeps the similarity to the anchor it was given: with other weights that number is another number (D17), and
here the vectors are in hand, so it is computed.

**Its baseline is the live history.** A shadow that kept its own history would need its own store and its own
write rules, and would answer "where would the catalog be". This one answers "what would the candidate have said
to this request, here", which is the question a change of policy raises on the day it is made, and needs no
state at all. The price is stated in the README: a revision the candidate would have refused still becomes the
baseline when the live policy stores it.

**A request's own threshold replaces the candidate's.** Were the candidate live, the request would have been
judged at the threshold it asked for. Judging it at the candidate's configured one would report disagreements
that the change of policy would never cause.

**`[shadow.policy]` is whole; `[shadow.fields]` and `[shadow.quantities]` are overrides.** A threshold belongs
to the weights and floors it was measured with (D3), so it is never inherited. Fields and quantities can only be
the live ones, since the candidate reads the live vectors under the live layout (D12), so they are written as
what changes, with `"off"` for no floor or limit as in a sweep grid (D29). `Config::shadowed` builds the
candidate's configuration from the live one, which is how the names and `required` are the same by construction.
`[shadow.model]` and `[shadow.qdrant]` are refused by name, with the reason.

**What it writes are results, flat, and their names are an interface** (D15, D17): `shadow_started` with a
`shadow_schema_field` and `shadow_schema_quantity` line per item, `shadow` and `shadow_reason` joined on `key`,
`shadow_summary`. They go to stdout, which a serving process did not write to before; that is why nothing is
written without a `[shadow]` table, and why the default is a line per disagreement and not per request. A
`shadow` line carries scores, verdicts, and codes. It names the record by id, as the request log line does, and
carries no text and no tenant: whether a tenant may be written to a log is the owner's decision, pending with
the telemetry plan. Fields can be added later; none may be repurposed.

**Agreement is not quality.** A disagreement says the policies differ, not which is right. The summary counts
both directions and nothing in the tool ranks a candidate by them. The ids are for labeling; the labeled pairs
go through `calibrate`, where a gate exists (`calibrate-diff --fail-on`).

**What it costs was measured, not argued.** One weighted comparison of vectors in memory per judged revision,
and three small clones per stored write so that the engine can still show the observer what it stored. Under
`load --server` on the development machine the medians with and without a shadow were within noise of each
other (README). One cost is not a number: a line is written under the record's lock by a blocking write, exactly
as the request log line is written to stderr. A stdout that nothing reads would stall a server with a shadow
the way a stderr that nothing reads stalls any server. Moving the write off the request is the model shadow's
queue, and is not worth having for a line per disagreement.

**A candidate model is not shadowed.** `docs/plans/18-shadow-evaluation.md` designs it: a second model, a second
collection, writes mirrored from the live decision, a lossy ordered queue, an import into the shadow collection,
a report. It is not built, for three reasons. The second model competes with the live one for the same cores
(Burn's CPU backend runs on the global rayon pool, so a low-priority thread bounds nothing), and what latency
that costs can only be measured on the production hardware and accepted by its owner. Whether a model is better
is answered on labeled pairs by `calibrate --model` and AUC (D24), with no service running; live traffic adds
only the cases nobody labeled. And no candidate model exists yet. The observer seam is what the model shadow
would plug into, so building the policy half first cost it nothing.

**Not built with it:** a `shadow-report` command (the summary line has the counts, `jq` lists the ids, and
intervals resampled by tenant need the tenant on the line), the shadow in `Describe`, and more than one shadow.
