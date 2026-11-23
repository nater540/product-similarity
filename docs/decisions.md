# Decisions

The choices that shaped this project, with the evidence behind them. Code comments refer to these by number.
Numbers quoted here were measured on an Apple-silicon laptop CPU with a release build; rerun `just throughput` and
`just calibrate` to reproduce them.

## D1. Burn 0.22 on the `flex` CPU backend, Metal behind a feature

Burn 0.22 changed how backends work: models are plain types, and a runtime `Device` selects the backend. `ndarray`
and `tch` are deprecated; `flex` is the pure-Rust CPU backend that replaces them. The workspace builds `flex` (with
`simd` and `rayon`) always, and the trainer adds Metal or CUDA with a feature. (The server had a Metal feature
too, until ONNX Runtime took over the forward pass and there was nothing left for it to speed up: D34.)

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
  brand-filtered search, snapshots, a dashboard for looking at what is stored, and headroom to millions of records
  without redesign.
- **What it was said to lack** was a compare-and-swap, and that was true of the Qdrant this was first written
  against. The one that is pinned now has a conditional upsert, and every write uses it (D35). A relational
  database does the same with a row lock or a version check.

So: Qdrant is a reasonable choice because neighbor search is wanted and it is pleasant to operate, not because the
scale demands it. If the organization already runs Postgres, a table with a `bytea` or `pgvector` column would do
the same job at this scale with one fewer service and with transactional writes. The store is behind a trait
(`VectorStore`, eight methods, one contract test suite), so that swap is a few hundred lines.

Layout in Qdrant: one collection, one point per record, two named vectors. `current` is HNSW-indexed and is what
searches match. `anchor` (see D8) is only ever read by id, so its HNSW graph is disabled (`m = 0`) and it lives on
disk. `brand_id` has a brand-aware keyword index. Point ids are the UUIDv5 of the record id.

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
while someone is still typing), and `FORCE` (always write: a human reviewer overrides the policy). A forced
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

Evaluating is read, compare, write. `Engine` serializes requests for the same id with a lock (at first one of
256 picked by a hash of the id; since D41 the record's own), so within one
server two concurrent revisions cannot both pass against the same baseline. Across replicas the locks do nothing,
and for a long time nothing else did either: this decision used to end by saying that callers had to serialize
edits to one record themselves, or route by id. Since D35 every write is conditional on the store still holding
what was read, which holds across replicas. The locks stay for what they are good at: they save a process from
racing itself, and they keep one record's requests in order for whoever is watching them (D30).

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

## D14. A brand, once set, must be named

A record may carry a `brand_id`: the key of the brand it belongs to. It is stored with the record and filters
searches. It is also checked: once a record has a brand, an `Evaluate` or `Delete` for its id must name the same
one. Naming another is refused, and so is naming none. The second half matters as much as the first: the mistake
the check exists for is a caller that has lost track of whose record it is editing, and a check that can be
skipped by leaving a field out does not catch that. A record stored without a brand takes the first one a revision
names, which is how a catalog indexed before brands were in use gains them.

This is a consistency check and not access control. The server does not authenticate anyone (see the README's
limitations), so a caller that can reach it can always supply the right brand.

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

## D17. A calibration is also data: flat events, intervals over brands

`calibrate` printed a report for a person and nothing else, and read only a four-key format that drops everything
a data export knows about a pair (its id, its brand, its split, whether it is a variant change). Real labeled
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

**Intervals resample brands, not pairs.** The two error rates and the operating points carry a 95% interval from
1,000 seeded bootstrap resamples. What is resampled is the brand, with all of its pairs: edits by one brand are
written by the same people about the same product line and are not independent, so resampling pairs would give an
interval that is too narrow, and narrower the more one brand dominates. That needs every labeled pair to name an
brand and at least two brands; otherwise no interval is printed, which is honest for a file that cannot support
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
with and without brands and quantities), `Delete`, and `FindSimilar`, with every rule of D7, D8, D14, and D16
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
it again, with no bookkeeping. The same request rule holds as anywhere else: an id stored under another brand
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
server's queue (D9). On a laptop CPU 2,000 four-field records imported in 19 seconds when Burn ran the model,
and import in 5 now that ONNX Runtime does (D34).

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
menu item is in order and needs no index. The name comes from a path the caller chose, so a property holds that an
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

**A third split, carved from the first.** `pairs carve` moves a share of one split's brands (by default from
`calibrate` to `validation`). Whole brands move, because pairs of one brand are not independent. Which brands is
decided by a hash of the brand's id and a salt, with no randomness and no state: the same command moves the
same brands, carving twice moves nothing more, and a larger share keeps every brand a smaller one moved. The
share has no default. It is not a threshold, but it is a judgment about how many labeled pairs a comparison
needs, and that depends on the data.

**Both binaries select by `split` alone.** The trainer trains on `calibrate`, compares on `validation`, and
reports on `holdout`; `calibrate --split` reads any of them. The alternative was a fold computed in memory from
a brand hash in each binary (which would allow k folds). One carved split was chosen because the two binaries
then cannot disagree about which pairs are which, and a file is something a person can look at.

**Catalog groups of measured brands are left out of training.** A catalog group has no split of its own. One
whose brand is in the validation or holdout split is skipped, or the brand held back for measuring would be
learned from by another route. A group that names no brand is kept and counted.

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

The match ranking, the match statistics, and the promotion gate joined the set when they were built: a match and
a promotion are decisions too. A run at `0210b39` tested 676 mutants in 25 minutes: 563 caught, 109 that do not
compile, 4 surviving. One was in the registry (a file that exists and cannot be read was not told apart from one
that does not exist, which would have recorded a first promotion over a damaged registry); three were in the
match statistics (what a score exactly at the threshold counts as, and that the two populations say which file
and split they came from). Each got a test, and each mutation was then applied by hand to see the test fail.

After the streaming import and the ONNX engine, a run at `a7b67c3` tested 681 mutants in 27 minutes: 571 caught,
110 that do not compile, none surviving.

After conditional writes (D35), a run at `9756a57` tested 689 mutants in 28 minutes: 575 caught, 113 that do not
compile, 1 surviving, and the survivor said something. With the per-record lock replaced by one that locks
nothing, every test still passed: the store's condition now gives the right outcome whether or not requests take
turns, so the test that used to catch a missing lock (thirty-two requests, one record created) no longer could.
What the lock still does is keep a process from racing itself, and that is what its tests now state: two
requests for one record each read it once, and thirty-two forced writes to one record all succeed with one read
apiece. The mutation, applied by hand, fails both.

After writes that outlive their request (D38), a run at `b1fc465` tested 701 mutants in 31 minutes: 570 caught,
118 that do not compile, none surviving, and 13 that ended at the tool's time limit, where there had been none.
A time limit is a catch, and a bad one: it means a test hung, and a test that hangs on a mistake hangs in CI
too. Eleven of the 13 hung in waits with no bound: the lifecycle tests' wait for a request to reach the
embedder (D37), and the new tests' waits for a write to reach the store, for every write to end, and for a
retry to be answered. Each wait now gives up with a sentence, and those eleven, run again at `315ce6f`, fail a
test. The other two make "no write is in flight" never true, so a stopping server never finishes: the tests
that say so fail in eight seconds, and the tests that only stop a server wait out its twenty-second drain,
which is longer than the tool waits. The embedding thread (`worker.rs`) is not one of the files, since it
decides no verdict, but it was run once for this change: of 100 mutants, nothing in the new queueing survived,
and five end at the time limit because they make the thread or the choice of a pass loop for ever.

After the audit (D42), `audit/compare.rs` joined the files: an audit decides no verdict, but what it calls the
same is believed, and a comparison that was dropped is an audit that passes. A run at `fd8010d` tested 757
mutants in 36 minutes: 624 caught, 133 that do not compile, none surviving, and none at the time limit. Of the
37 in the new file, 35 are caught and 2 do not compile.

After the backup of anchors (D43), the restore's rule is in `engine.rs`, which was already one of the files,
and `anchors/file.rs` joined them: what a backup reads back is written into records. A run at `65742af` tested
811 mutants in 40 minutes: 663 caught, 148 that do not compile, none surviving, and none at the time limit. One
had survived a run over the change alone while it was being written: a restore that kept the record's old time
of writing, which now has a test.

**Why an allowlist of files.** The alternative, excluding what should not be mutated, grows a list of everything
whose only tests need the model or Qdrant, and a new file is mutated by default whether or not its tests can run.
The files listed are where a mutant that survives means a wrong decision.

**Why one fixed proptest seed.** The property tests draw a new seed each run. A mutant that one seed catches and
another misses would make the survivors differ from run to run, and a gate that changes its mind is ignored. The
recipe fixes the seed and turns off the regression files, which would otherwise be written into a copy that is
thrown away.

**Why it is not in `just ci`.** Twenty-five minutes of twelve cores. It is run by hand after a change to what
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
That was with Burn running the model. The README's table is the same test with ONNX Runtime (D34): 627 to 2,448
edits a second, a median from 1.6 ms to 22 ms, and creates leveling off at about 400.

**The loop is closed, and says so.** Each caller waits for its answer before sending again. That measures latency
at a fixed number of callers. Latency at a fixed arrival rate, where a server that falls behind makes its queue
grow, is a different measurement and needs an open loop; the README warns against reading one as the other.

**The store defaults to memory, and Qdrant is never the default.** A load test that wrote into the configured
collection because nobody said otherwise would be a bad afternoon. Against a running server the records go under
a brand of their own and are deleted afterwards.

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
`shadow` line carries scores, verdicts, and codes, and no text. It names the record by id and by the brand the
request named, as given: the owner decided on 2026-11-11 that a brand is an identifier the server may write,
which is also the answer the telemetry plan was waiting for. A request that names no brand gets a line without
the key. Fields can be added later; none may be repurposed.

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

**`shadow-report` reads the file back.** Totals come from the summaries (each counts from its server's start,
so the last of each run is taken and the runs are added up; `run` on the lines is what tells them apart), and
everything about which revisions comes from the `shadow` lines. The two are never mixed: a rate is over what
the summaries counted, and an interval, resampled by brand, is given only when the file has a line for every
judged revision (`events = "all"`), because only then is each brand's share of the judged revisions known. The
reasons listed for a disagreement are the ones the rejecting side gave. It writes the disagreements as a list
to label and prints, last, that a disagreement is not a verdict on the candidate. Its numbers are in
`offline::stats::shadow`, like every other report's (D17).

**Not built with it:** the shadow in `Describe`, and more than one shadow.

## D31. A config names the model its thresholds were measured with, and a gate records what was put into service

D3 says thresholds do not transfer between models, and D12 that a new model needs a new collection. Both were
rules a person had to remember. Three small things turn them into checks.

**`model.expected_fingerprint`, opt-in.** A config may name the fingerprint its thresholds were measured with.
With the key, `serve` and `import` refuse to run with any other model; the commands that measure (`compare`,
`calibrate`, `calibrate-drift`, `load`) warn and go on, because measuring a new model under a copy of the old
config is exactly how one is evaluated. Without the key nothing is checked, and `serve` and `import` say so in
one warning that carries the fingerprint to copy. The owner chose opt-in on 2026-11-11: `config.example.toml`
does not set it, so no existing config changes behavior, and a production config that sets it cannot be pointed
at another model by an edit to `model.dir` alone. It is an assertion and not an input: no part of the
fingerprint or the layout id (D12), and it changes no vector.

**The server does not read a registry.** The registry is a local file that names local paths; a container does
not have it, and a server that needed it could not start without it. What has to hold at start-up is that the
config and the model agree, and that travels in the config.

**The registry is a file of entries, appended and never rewritten.** `models/registry.ndjson`, beside the model
directories and never inside one (D18), ignored by git with them. `registered`, `promoted`, `retired`. It is a
data file the tool owns, like `provenance.json`, not a result stream, so it is not written through
`similarity-log` and D17's flatness (which `tracing`'s scalar fields force) does not apply: a `promoted` entry
nests the config it was judged under and a reference to its calibration. Its names are an interface all the
same. A line that is not an entry, or is of a newer format, is an error naming the line, not something guessed
at. An append is one write of one line, synced. Two promotions at the same instant both compare with the same
model in service; that is accepted and not locked against.

**There is no `calibrated` entry.** The suggestion had one, holding a calibration's numbers. The calibration
file, or its run directory (D23), is the record; a copy of its numbers would be a second source. A promotion
points at its calibration by path and SHA-256 and carries only the counts the gate compared, from the same
`Calibration` the report is rendered from, plus the AUC.

**A model is keyed by digest, a promotion by fingerprint.** One directory has one digest and as many
fingerprints as there are widths and token limits to load it at. The model in service is the latest promotion of
a model not retired since, so retiring it puts the one before it back, which is what a rollback is. A model is
retired with every width it was promoted at; going back to another width of the same model is a promotion.

**The gate measures and chooses nothing** (D24). It never suggests or writes a threshold: the suggestion's
"prints the thresholds from that calibration" would have made the holdout a validation set. It judges the
candidate's holdout pairs again under the candidate's own config (`diff::rejudge`), so what is promoted is the
config, whatever thresholds the calibration happened to be run with. Four things are refused whatever is said:
a calibration of another model than the config loads, one not restricted to the `holdout` split, a holdout
without both labels, and a config whose fields are not the calibration's.

**Each model is held to its own thresholds.** With a model in service, its calibration is judged under the
config it was promoted with, which is why the entry keeps that config, and the two are compared over the same
pairs (`diff::same_pairs`, or the comparison is refused and `--baseline` asked for). D24 rules out judging a
model at another model's threshold; this does not do that. Counting errors under each model's own policy is
the question a promotion asks ("will this config do worse in service than the one there now"), and the report
prints the AUC of each beside it, which needs no threshold. The default refusal is `errors-up`: counts, not
intervals. The intervals are printed, so a person can see when one pair decided it. `--override` with a reason
passes a failed comparison and a retirement, and the reason is in the entry.

**The recorded baseline has to be the file it was.** The model in service is compared through the calibration
its promotion recorded, checked against the recorded hash. A calibration given by hand (`--baseline`) is a
fresh measurement and is held only to being of that model on the same pairs.

**The collection is named, not created.** A candidate with the layout in service keeps its collection (D12:
thresholds and weights need no re-index). A new layout whose config still names the collection in service gets
that name's stem and twelve hex digits of the layout's hash; a new name the config already has is kept. The
gate prints the name and records it. It cannot know what the operator then does.

**Not built:** a minimum holdout size. The plan proposed warning below 40 recycles, a number nobody measured.
The report prints each rate with its interval instead, which says the same thing without a constant.

## D32. Matching a menu item to a record is a read, scored exactly, with a threshold nobody has measured yet

The same model and the same stored catalog can answer a second question: not "is this edit still the same record?"
but "which stored record is this menu item for?". A menu item is an entry on a retailer's menu, in the retailer's
words, often with only a name. `FindSimilar` with an ad hoc record already half-answers it, and D5 says why only
half.

**The search score is the exact score times a factor, and the factor is known.** For a menu item `Q` and a stored
record `C`, the cosine of the two composites is the weighted mean over the fields both have, multiplied by
`W(Q∩C) / sqrt(W(Q) * W(C))`, `W` being the sum of the weights of a set of fields. The factor depends only on
which fields each side filled in. A name-only menu item against a record with all four example fields is scaled by
`4 / sqrt(4 * 10) = 0.63`; against a record with only a name, not at all. So a search ranks the record that
resembles the menu item in *which fields it has* above the one that resembles it in *what they say*. A property
holds the formula (`the_search_score_is_the_match_score_scaled_by_the_field_overlap`), and the README's example
shows it on the real model: 0.940 exact, 0.595 in the search.

**So a match searches, then scores what it found.** `matching::rank` in `similarity-core` scores each candidate
with `score::compare`, stored record as baseline, menu item as candidate, no anchor: a field only one side has is
reported and left out. The engine and `calibrate-match` both call it, so the server and the measurement rank
alike. A match is a reading, not a revision: no floor and no quantity limit applies, and none is ever reported.

**The over-fetch is a cost, and it is measured.** Only what the search returns can be scored, so `Match`
fetches `matching.overfetch` records for each candidate it returns, capped at 1,000. The suggestion said five.
Five is a guess, kept as the default and not defended: `calibrate-match` scores every product exhaustively as
well and reports how many menu items got another ranking, which on a real catalog is what says whether five is
enough. `VectorStore::search_vectors` exists so that the re-rank is one round trip and not one `get` per
candidate; both stores build both searches from one request, and the shared contract holds them to the same
results in the same order.

**A new RPC, not a mode of `FindSimilar`.** `FindSimilar` documents its similarity as the composite cosine and
returns ids. `Match` returns another score, per-field and per-quantity detail, and a decision. Changing what
`FindSimilar` reports would silently change a documented number for its callers.

**A menu item is mapped onto the product schema and has no layout of its own.** Another set of fields is another
layout (D12), and a menu item under one could not be compared with a product at all. The caller maps the menu item's
columns onto the schema's names, as it does for products, and leaves out what has no counterpart.

**A menu item carries no id and no brand, and a request that sends either is refused.** Both belong to the system
the menu item came from. Accepting a brand on the menu item would invite a caller to pass the retailer where the
brand is meant. The search is narrowed by the request's own `brand_id`. Without one every brand's records are
candidates: the owner's decision of 2026-11-11, and what `FindSimilar` has always done. As everywhere, the
brand is a consistency check and not access control (D14).

**It never writes.** No menu item is stored, no record changes, and D7, D8, and D20 are untouched by construction.
The property state machine sends `Match` requests among the others and checks the store is exactly what it was.

**Quantities are reported and decide nothing** (D16). A 3.5 g menu item for a 1 g product may be the same product
in one catalog and another product in the next. Each candidate's quantities are compared and returned, and
`calibrate-match` counts how the menu items' agree with their own products'. Whether one should stop a match waits
for the answer to that question about the real catalog. Containment (a batch's potency inside a product's range)
would be the right comparison for some of them, and is not built.

**The threshold separates two populations, and the second is a stand-in.** The first is what menu items score
against their own products. The second should be what a menu item scores when its product is not in the catalog,
which the groups cannot contain: every menu item in them has a product. The best product that is *not* its own is
exactly what such a menu item would be matched to, so it stands in. These are the two populations a revision
threshold separates (D17), in another guise, and `stats::matching::populations` writes them as the `edit` and
`recycle` pairs of a `Calibration` so that the suggestion, the operating points, the AUC, and the
brand-resampled intervals are the tested ones and not a second copy. What the stand-in cannot give is the base
rate: how many real menu items have no product. The data request now asks for it.

**No threshold is shipped.** `matching.threshold` is a measurement like any other (D3), nobody has made it, and
without one `Match` ranks and says `NOT_JUDGED`. The sample groups are synthetic and no number may be read from
them. The decision is on the best candidate's score alone; `margin` is reported, so that a rule on the gap to
the second can be judged later from the events without scoring anything again.

**Held-out groups.** A group carries a `split`, and the trainer leaves out a group in the holdout or validation
split as it leaves out a group whose brand has pairs there (D24): a match threshold measured on groups the model
was trained on would flatter it. `--split` selects whose menu items are scored and never which products are in the
index, since a menu item has to be found among everything it could be confused with.

**Not built:** separate weights for matching (the `match_field` lines carry what would be needed to judge them
without a model), a `calibrate-match` diff, storing menu items, and any use of quantities in the ranking.

## D33. New records can be streamed in; a stream only creates, and shares the model a batch at a time

"No bulk RPC" was a stated limitation: a caller that cannot run the `import` command beside Qdrant loaded a
catalog one `Evaluate` at a time, and got batching only by the accident of other callers arriving together.
`Import` is a bidirectional streaming call: records in, one answer for each out, in order.

**It is the import, over a wire, and so it only creates** (D20). The call ends in `Engine::import`, the code the
command runs. A record whose id is stored is answered `EXISTS` and left exactly alone, so a stream cannot
replace a baseline, cannot move an anchor, and cannot be used to get an edit past the policy: a change is an
`Evaluate`. Creating only is also what makes a stream safe to cut off and send again, with no cursor to keep and
no "how far did it get" to answer. A judged batch (many revisions in one call) is a different thing with
different rules, and is not this.

**A record that is stored is not embedded.** `Engine::import` used to embed every record of a batch and only then
look at the store, so sending a file again cost as much as sending it the first time: 29 seconds for 3,000
records that were all there. "Send it again" is the resume, and "send everything, the new ones will be created"
is the easiest thing for a caller to build, so that had to be cheap. It now looks first, outside the record's
lock: a record that is there costs that read. A record that is not there is embedded, and whether it is stored
is decided again under the lock when it is written, so the early look decides nothing by itself. The offline
command gets the same saving.

**One record failing does not end the stream.** The answer carries the status code the record would have got from
`Evaluate` and the sentence. A caller loading fifty thousand records must not lose the stream to the one with a
field the schema lacks. The call's own status is the transport's; everything about a record is in its answer.

**A stream is batched by its own backlog, in small batches.** The loop takes the first record waiting and
whatever else has already arrived, up to `server.import_batch_records`, and never waits for more: the worker's
own rule (D9), one level up. A trickle makes batches of one. What the setting trades was measured, not guessed.
A batch is one job on the one embedding thread, and every other request waits for the job in front of it. At 8
records (about one forward pass of four-field records) a stream runs at 81 records a second, the ceiling 64
concurrent callers of `Evaluate` reach, and an edit by someone else beside it takes a median 86 ms instead of 6 ms.
At 1 the stream runs at 48 a second and the edit takes 25 ms. The default is 8: a save of 86 ms is still a fast
save, and a bulk load is the less frequent event. It is a setting because that is a judgment about someone
else's callers. (Those figures are from when Burn ran the model. With ONNX Runtime, D34, the stream runs at 398
records a second at 8 and the edit beside it takes 17.5 ms instead of 1.6 ms; at 1, 220 a second and 5.5 ms. The
trade has the same shape and a fifth of the cost.)

**A full queue is waited out.** A single request that finds the model's queue full is refused at once, so that
it can back off (D9). A stream is its own back-off: it holds its batch, waits (10 ms, doubling to 250 ms), offers
it again, and meanwhile reads nothing, which the transport turns into back-pressure on the sender. Both channels
around the loop hold one batch, so nothing is buffered without bound. After ten seconds without room the batch's
records are answered `RESOURCE_EXHAUSTED` and the stream goes on.

**A cancelled stream loses nothing and leaves nothing half done.** The loop runs in its own task. Each record is
created whole or not at all (one `put`), what was created stays, and the rest was never taken from the stream.

**Where the code is.** `import_stream.rs` is the loop and knows no protobuf: records in on a channel, answers
out on a channel, so its batching, its waiting, and what it does when nobody listens are tested without a
socket. `grpc.rs` converts and spawns. The engine is unchanged apart from the early look.

**Measured through `just load`,** which gained a `stream` phase so that the figure is produced like the others.
The trade-off table in the README is `load --server` beside a stream that never ran dry.

**Not built:** a streaming or batch `Evaluate`; per-brand fairness between a stream and other callers (feature
27), which this makes more pressing; a final summary message on the stream (the caller counts its answers, and
the server logs one `import` line).

## D34. ONNX Runtime runs the forward pass; Burn stays the definition of the model

Asked how this would compare with a Python or a Node service, the honest answer had to be measured, and it was
not the expected one. On the same machine, model, and texts, the Burn CPU backend had the slowest forward pass
of everything tried: PyTorch was up to three times faster, and ONNX Runtime, which is what a Node service would
call, up to five times. The language around the model hardly matters; nearly all of a request is the forward
pass, and every stack hands that to native code. What this service had over the others was its queueing,
its memory, and one stack for training and serving, not its arithmetic.

**What was tried, in the order of the owner's preference.** Pure Rust first. Candle was no faster than Burn.
Tract was two to three times faster on one core, but could not analyse the exported graph with variable input
sizes and needed a plan per size. Thread count was not the cause: Burn takes the same four milliseconds for a
short field on one core as on twelve. Then ONNX Runtime through the `ort` crate, which the owner had taken for a
Rust port and accepted for its speed once told it is a binding to C++, on condition that the gRPC service stays
a Rust binary. It does.

**One engine for everything the server binary does, and no setting for it.** For its first day this was a
choice: `engine = "onnx"` in the config, in a binary built with `--features onnx`, beside a default of Burn. On
2026-11-12 the owner dropped Burn as a way to serve: it had been expected to perform better than it does, and
keeping it was not a requirement. A second route cost a setting, a build feature, two sets of measurements, and
a shipped config whose default was the slow one. So `model.engine` is gone, and with it `model.device` and the
server's `metal` feature, which only ever chose how Burn ran. A config that still has either line is refused
with a sentence that says what replaced it, not as an unknown key. Every command of `record-similarity` that
embeds (`serve`, `import`, `load`, `compare`, the `calibrate` family, `model promote`) loads the model the same
way and needs the exported file. `Embedder` in the core still has both engines; the tokenizer and its D21 rule,
the length buckets, truncation to `output_dim`, normalization, the worker thread and its batching are shared.

**What Burn still does.** It defines the model: the encoder, the strict loader, the fingerprint. It trains it;
the trainer never links ONNX Runtime (D13), which is why `onnx` stays a feature of `similarity-core` that the
server always turns on and the trainer never does. It is what every load is checked against (below). And it is
what the tests run: the tiny encoders of `cargo test`, and the real-model tests (parity with PyTorch, the pinned
verdicts of D22, the full stack against Qdrant), which load with `Embedder::load`. The pinned verdicts are
statements about the model, and the engine is held to the model at every load, so they are not made a second
time under the engine; `just test-onnx` is what holds the two to each other and to PyTorch. That is also why CI
needs no Python.

**What that costs.** A model directory is not usable by the server binary until it has been exported, which is
Python, once for each directory: the fetched model, a fine-tuned one, and any checkpoint that is to be
calibrated. The trainer's own validation scores are computed in Burn and need none. The server no longer builds
without the ONNX Runtime download, and on Linux not without the C++ standard library of GCC 13; there is nothing
to fall back to. The container image exports the model in a build stage of its own.

**One model, two engines, one fingerprint.** The graph is exported from the directory's weights and computes
the same function in another order of operations. On the pinned model the engines differ by at most 2e-7 in a
component of a unit vector. So the fingerprint and the layout id do not say which engine made a vector (D12):
they say what the vector means, and that is the same. Thresholds are measurements of the model (D3), not of
the engine.

**The exported file is checked against the weights at every load.** This is the decision the rest depends on.
`ModelFiles::digest` hashes what the Burn encoder reads; under the ONNX engine the numbers come from another
file, which the digest does not cover and which goes stale silently when a directory is fine-tuned or an `onnx`
folder is copied from another model. A sidecar recording which weights the graph was made from would be trusted
on its word. Instead the loader builds the Burn embedder as always, embeds four texts in one batch with both,
and refuses the graph if any component differs by more than 1e-5, fifty times what the engines really differ
by. The Burn encoder is then dropped. It costs a tenth of a second at start-up, and it means a wrong graph
cannot serve one request. A real test damages a copy of the weights and expects the refusal.

**The export is Python.** PyTorch's exporter is the tool that turns a PyTorch module into a graph, and writing
a second one in Rust would be work with no benefit. The owner relaxed the rule on 2026-11-12: Python where it is
the best tool, not for everything. `tools/export_onnx.py` exports the sentence-transformers pipeline up to, and
not including, normalization, checks the result against PyTorch on batches of other shapes than it was traced
with, and writes the file by rename. Nothing at run time needs Python.

**Where the C++ comes from: the prebuilt download, for now.** The `ort` crate's `download-binaries` feature
fetches a prebuilt static library at build time from the crate author's distribution. That is convenient and it
is a third party's binary inside the server. The owner chose to keep it for now (2026-11-12). The crate is
pinned to an exact version, and the alternative (`load-dynamic`, with a runtime library the organization builds
or takes from Microsoft's releases) is a change of features, not of code, so the choice can be taken again
without touching the engine.

**What each platform needed.** On macOS the prebuilt library has Objective-C that calls into the compiler's
runtime library, which Rust does not link unless told (`.cargo/config.toml`); and ONNX Runtime hands graphs to
CoreML by default there, which was slower and failed on a batch, so the session is told to prefer the CPU. On
Linux neither applies, and something else does: the prebuilt library is compiled against the C++ standard
library of GCC 13, so it does not link on Debian 12 (the first container build failed on exactly that). The
image moved to Debian 13, which the owner accepted for production on 2026-11-12.

**Measured.** On macOS, `just throughput`: a four-field record in 4.2 ms against 20.2 ms. `just load`, through
gRPC: one edit in 1.6 ms against 5.6 ms, and a ceiling of about 2,400 edits a second against 470, with the
queue, the batching, and the single embedding thread unchanged. An `Import` stream runs at 404 records a second
against 80. **In a Linux container on the same machine the gain is 1.7 to 2.4 times, not 4 to 5:** one edit in
3.5 ms against 6.1 ms, 822 edits a second at 16 callers against 346. Burn is about as fast there as on macOS;
ONNX Runtime's macOS build is the one that does better on that hardware. The engine is worth having at twice
the speed, and the first figure should not be quoted for a Linux deployment. The Burn figures in this paragraph
are the record of why Burn stopped serving; the README no longer carries tables a reader cannot reproduce.

**Not done:** x86-64, which is unmeasured and is probably what production runs on; several embedding threads
(ONNX Runtime may make one thread per core a better design than one thread across cores); an exporter in Rust,
which would take Python out of the image build and out of the fine-tuning loop; pinned versions for what the
export script installs.

## D35. Every write is conditional on what was read

Two replicas handed two edits of one record at the same instant could both read the same baseline, both accept,
and both write: the second write replaced the first unseen, and an edit that should have been judged against
the other was not. D6 and D10 said the store could not prevent it. The Qdrant that is pinned can.

**The store writes only over what the writer read.** `VectorStore::put_if` takes what the writer expects to
find: nothing (`Expected::Absent`), or the version it read. It writes if that is still so and says whether it
did. `delete_if` is the same for a delete. The engine uses nothing else: a first version is written only if
there is still no record, a revision only over the version it was judged against, a forced write the same (its
revision number would otherwise be wrong), an import only if there is still no record, and a delete only of the
record whose brand was just checked.

**A writer that lost has written nothing, and judges again.** It reads the record as it is now, embeds what
differs from that, and judges and writes against that, up to four times (`WRITE_ATTEMPTS`). So a request that
arrives together with another gets the answer it would have got had it arrived second. After four losses in a
row it is answered `ABORTED`: nothing was written and the request can be sent again. That takes another writer
changing the same record in the moment between this one's read and its write, four times running, which is not
contention but something rewriting a record in a loop. A request that writes nothing (a dry run, a rejection,
the same record again) is answered from what it read and has nothing to lose a race with.

**A version is the revision and the time together.** The revision goes up by one with every write, so within a
record's life it names one write. A record that is deleted and stored again starts over at 1, and a writer
still holding the old record's revision 1 must not be taken for one that read the new record. The time of the
write tells the two lives apart unless both were written in the same second, which leaves this: a record
created, deleted, and created again within one second, while a request that read the first is still on its way.
A mark of its own for every write would close that too, at the price of a field on every stored record that
every caller would have to carry. It was not worth it.

**In Qdrant.** A create is an upsert in insert-only mode. A revision is an upsert in update-only mode with a
filter on the stored layout, revision, and time: update-only, because a filter alone would let the write through
as a new point if the record had been deleted in between. Qdrant answers `completed` whether or not the
condition held, so each conditional write puts a mark of its own in the payload and reads the point's payload
back: it holds that mark only if that write stored it. A delete is a delete by filter, and then a look. The
contract test runs the same assertions against the in-memory store and the real one, including eight writers at
once that all read the same version (one wins, and it is the one that was told so) and two connections racing
for twenty rounds.

**Held as a property.** Two engines on one store and no shared lock; any history; then any two requests, with
the second let in between the first one's read and its write. Both answers and what is stored must be what the
two requests give one after the other, in one order or the other. One answer may differ, and the property names
it: two deletes of one record at once are both told that there was a record, where one after the other the
second is told there was none. The store can say that a record is gone and not which of two deletes removed it;
telling them apart would take a claim written before the delete and two more round trips, for an answer nobody
acts on.

**What the read-back cannot see.** A writer whose write was applied, and replaced by another writer before it
looked, finds another mark and concludes it lost. It judges again against the newer record. The result is that
of the request having been sent twice, which is something a caller that retries does anyway.

**"An import only creates" is now the store's promise** (D20), not one process's lock: the `import` command can
run beside a server that is taking edits, and a record either of them stored a moment before is left alone by
the other.

**The locks stay** (D10). Within a process they keep two requests for one record from racing at all, which is
cheaper than losing and judging again, and they keep the order in which a record's requests are observed (D30).

**What it costs** is one more read of a small payload for every write. Through gRPC against a local Qdrant, one
caller, medians of two runs before and two after: a create 5.8 ms and then 6.0 ms, an edit 2.9 ms and then
3.1 ms, a save that changes nothing 0.26 ms both times. With 16 callers editing, about 1,700 edits a second
became about 1,400: the model serves a batch at a time, and the store's share of each request grew. On the
in-memory store nothing changed.

**Upgrading.** Nothing stored changes shape: the mark is one more payload key, a point written before has
none, and its revision and time are what the condition reads. But a replica from before this change still
writes unconditionally, so two replicas coordinate only once both are this version.

**Not done:** a Postgres store, where the same contract is a version check in a transaction; an index on the
two payload keys the condition reads, which a Qdrant run in strict mode may ask for before it accepts the
filter.

## D36. The words are the ones the owner's company uses

The first version of this service was written in general terms: a record has a *tenant*, a labeled pair has an
*owner*, and `Match` is given a *listing*. Where the owner works those are the wrong words. A record's tenant
and a pair's owner are both the brand that owns the product. And a "listing" is a dispensary: it owns menus,
which hold menu items, and a menu item is what `Match` is given. A reader there met "listing" in the API and
thought of a store.

**What they are now** (the owner's choices, 2026-11-12):

| Was | Is | Where |
|---|---|---|
| `tenant` | `brand_id` | `Record`, every request and answer, the import export, the stored payload, the `shadow` events, `--brand-id` |
| `owner` | `brand_id` | pair, history, and group files; the calibration and match events |
| `listing`, `listings` | `menu_item`, `menu_items` | `MatchRequest`, group files, the `match_*` events |
| `merchant-1` | `brand-1` | the examples |

`brand_id` and not `brand`, because `brand` is taken: it is the text field that holds the brand's name, which is
scored. The key is an identifier, compared exactly and never embedded. Renaming the text field instead would
have changed the set of field names, and with it the layout id of everything stored (D12). In prose the key's
owner is simply "the brand".

Two words were offered and kept: `catalog`, for the stored product of a group, and "retailer", for whoever
owns a menu.

**Everywhere, with nothing read under the old names.** The alternative was to accept both, which leaves two
spellings in every file format for good. Nothing was in production. So:

- The `.proto` renames the fields and keeps their numbers, which is all a client built from the old file
  depends on, and reserves the old names so that they are never given to something else.
- A file format that refuses unknown keys refuses the old names by that rule (`tenant` in an import export).
- A format that ignores unknown keys would have read an old file as one with no brands, without a word: pairs
  that no longer split by brand, intervals over nothing, held-out brands trained on. Those formats refuse
  `owner` and `listings` by name and say what each is called now.
- A stored point that still has `tenant` is refused when it is read. Decoded as it is, it would be a record that
  belongs to no brand, which any request may edit or delete (D14).

**Event names moved too**, against the rule that they never do (D17, D30, D32): `tenant` and `owner` are
`brand_id` in every event that carried one, `owners` is `brands`, `shadow_report_tenant` is
`shadow_report_brand`, and `match_listing` is `match_menu_item`. The rule exists because other tools read these
names, and no other tool reads them yet. This is the one time; the rule stands.

**What was left alone.** The plans under `docs/plans/` are records of what was proposed and say so at the top.
The curated cases and their pinned verdicts hold no brand and no menu item, and did not change.

## D37. Live and ready are two answers, and a server leaves the rotation before it stops

The server said `SERVING` once, before its first request, and never anything else. A replica whose Qdrant had
gone away kept saying it, and a load balancer kept sending it requests to answer `UNAVAILABLE`. And when it was
told to stop it closed its listener at once, so the last requests of every rolling deploy raced the close.

**Two questions.** *Is the process up?* is the health service's empty service name, and the answer is yes for
as long as it runs. *Could a request be served now?* is `recordsimilarity.v1.RecordSimilarity`. They are kept
apart because they call for opposite things: a process that is not live should be restarted, and a process
that is not ready should be left alone and sent nothing. A replica whose store is down is the second kind.
One answer for both would have an orchestrator restarting healthy processes for as long as Qdrant was away.

**Ready means two things hold** (`Engine::check`): the embedding thread is alive, and the store answered the
last time it was asked. The store is asked every `server.store_check_every_ms` (2 seconds), with a call that
reads no record: for Qdrant, whether the collection exists, which also catches a collection deleted under a
running server. A check that takes longer than the interval counts as no answer. "The model has answered a
warm-up pass" needed nothing: the server does not listen until it has. The first answer is found out before
the listener can be asked, so a server that starts with its store down has never said ready.

**Not ready is advice, not a gate.** A request that arrives anyway is answered as well as it can be. Refusing
it would make the health check a second thing that can be wrong about the store, and the store's own error is
the better answer.

**One failed check is enough, and so is one that succeeds.** No counting of failures before the answer
changes: the thing that reads this answer (a kubelet, a load balancer) already has a threshold of its own, and
two of them multiply into a delay nobody chose.

**Stopping has an order.** On `SIGTERM`: not ready at once and still live; serving for
`server.drain_notice_ms` (5 seconds) so that whoever routes requests can act on that; then the listener
closes; then the requests in flight get `server.drain_deadline_ms` (20 seconds); then the process exits, with
a warning if it left any behind. The two defaults add up to less than the 30 seconds Kubernetes gives a pod.
They are not measurements and not thresholds: what the notice should be is how long a deployment's load
balancer takes to act, which only the deployment knows.

**Ctrl-C is not `SIGTERM`.** A person at a keyboard has no load balancer to wait for, so Ctrl-C stops without
the notice. And a second signal of either kind ends whichever wait the server is in: asking twice means now.

**What is lost at the deadline is nothing that was promised.** A request cut off there was never answered. A
write is one conditional step in the store (D35), so it happened or it did not, and an `Import` stream that
ends early is sent again (D33).

**Checked against the real thing.** With Qdrant stopped and started under a running server, the health
service went `SERVING`, `NOT_SERVING` (and live throughout), `SERVING`. With the notice at 1.5 seconds, the
server exited 1.60 seconds after `SIGTERM` and 0.09 seconds after Ctrl-C.

**Not done:** a count of the requests left behind at the deadline (that is a metric, proposal 26); taking a
failing request as a sign that the store is down, sooner than the next check; anything on a cluster.

## D38. A request can be dropped anywhere it waits; a write never is, and the model works only for callers that are waiting

tonic applies a client's `grpc-timeout` to every handler and drops the handler when it runs out, and drops it
when the client hangs up. Checked in its source (0.14.9, `transport/service/grpc_timeout.rs`): the timer and
the handler are polled together, and the handler is gone the moment the timer fires. So every `.await` in a
request is a place where the request can end. Two things followed from that, and neither was handled.

**The model worked for callers that had left.** A request waiting for the embedding thread is a job in a queue
with a channel to answer on. The thread never asked whether anybody held the other end. Under overload, which
is when callers give up, it embedded every abandoned request in turn, and by the time it reached one that was
still wanted, that caller had given up too. Measured, with 256 callers each waiting 50 ms for an edit (twice
what the model can do): 98 and 105 of 20,000 edits answered in time, 14 a second, from a server that does
2,450 a second when nobody is in a hurry. At 128 callers, just past its capacity, 2,061 and 2,152.

**A write could be cut in half with nobody knowing.** A handler dropped while `put_if` is on its way to the
store leaves a write that may or may not have been applied. The caller was told nothing, the observer was told
nothing, the request's log line was never written, and the baseline may have moved. The README tells the
caller to keep its database and this service in step, and this was the one case where the caller could not
know that it had to.

### The model

**A job whose caller has gone is passed over.** The reply channel says so (`is_closed`), and the thread asks
before every pass, of every job that is waiting. That costs nothing and needs no deadline: it covers a cancel
as well as a timeout.

**A request that says when its caller stops waiting is held to that.** The gRPC layer reads `grpc-timeout`
and scopes the handler with `worker::by`, a task-local, which the worker reads where it queues texts. A
task-local and not an argument, because the deadline is a fact about the request and would otherwise be a
parameter of every call between the handler and the queue, and of every stand-in embedder in the tests.

**A pass is made of the requests it can answer in time.** A pass answers everybody in it when it ends. The
thread remembers the least time per text any successful pass has taken; a pass cannot end sooner than its
texts take at that rate. A request is in a pass only if the pass, with it in, can end before its deadline,
most time left first, which lets the most in. A request whose deadline has passed, or will have before the
pass about to run can end, is told `DEADLINE_EXCEEDED` then and not embedded.

**The bound is one-sided on purpose.** It is the fastest the model has been, so a request is refused only for
a deadline that cannot be met, never for one that merely might not be. An average would refuse requests that
would have been answered, and a request refused wrongly is a failure this server caused. A request alone in a
pass is never refused on a guess at its own texts at all: the bound comes from other texts, and with nobody
else waiting there is nothing to gain. So an idle server only ever answers `DEADLINE_EXCEEDED` for a deadline
that has passed.

**In order, until a request is lost; then the newest first.** This is the part that was found by measuring.
With everything above and requests served in the order they came, a server at twice its capacity answered
1,950 edits a second in time, and the median answer arrived 42 ms into a 50 ms deadline. In order, the request
being served is the one that has waited longest. Once more arrive than can be served, the queue grows until
that request is at the very edge of its deadline, and stays there: every pass is made of requests with just
enough time for the model by the bound, and none for anything a request does next. So the thread serves in
order only while nobody is being lost. The moment a pass loses a request (gone, out of time, or refused by the
bound), it is filled from the newest end instead, and so is the next, until a pass is put together with nobody
lost. The newest have the most time left; the oldest, which cannot all be served anyway, are the ones that are
lost. The median answer then arrives at 24 to 26 ms. A server that is keeping up never leaves the order they came
in, and a caller with no deadline cannot make it leave.

**The bound on waiting requests counts requests that are still wanted.** `model.queue_depth` was the length of
a channel. It is now a count that a request leaves when it is taken into a pass, refused, or dropped by its
caller, whichever is first. A queue of abandoned requests used to refuse every new one until the thread had
worked through it (65,000 refusals in seven seconds, in the measurement above); now it refuses nobody.

**Nothing is refused on arrival but a deadline that has passed.** An earlier version refused a request when
the texts waiting ahead of it could not be embedded in time. Once the newest are served first, "ahead" is not
where the other requests are, and that test refuses exactly the requests that would have been answered.

**Measured again**, same load, same client, a store in memory: 2,319 and 2,320 edits a second answered in
time at 256 callers, and 2,364 and 2,374 at 128, of 2,450 unhurried. (Out of 20,000 that is 7,557 and 7,561,
and 12,780 and 12,788: fewer than in order, where it was 15,000 at 128 callers, because callers that are
answered in 25 ms come back sooner than callers answered in 40. The rate is the measure.) In an
in-process run at 256 callers the model embedded 7,859 texts for 7,565 answers in time. Without a deadline
nothing moved (the README's table was run again: 2,448 edits a second at 64 callers against 2,435 before).

**What it does not do: anything about the store.** All of this decides who gets the model. Against Qdrant in
Docker on the same laptop the store is the limit, about 1,500 edits a second whatever the number of callers,
with a median of 64 ms at 128 callers and no deadline at all. At a 50 ms deadline 209 of 10,000 edits were
answered in time at 128 callers and 80 at 256 (before: 41 and 68). The requests were embedded promptly and
then waited on the store, where nothing limits how many do or says who goes first. Their writes were finished
and logged, which is the other half of this decision, and a retry is answered from what was stored. A limit on
writes in flight, with the same rule for who is let in, would be the same idea one stage later. It is not
built, and it should be measured against a store that is sized for the job before it is.

**Also not done.** The bound is the fastest pass since the process started, so in a server's first moments,
before a full pass, it is rough. A request with no deadline that was already waiting when the server fell
behind waits until it has caught up: newer ones are served first for as long as requests are being lost. And
`Import` streams carry no deadline: a stream is one long call, and its batches are already paced by the model
(D33).

### The write

**A write runs to its end on a task of its own** (`Engine::committed`): the store call, then the observer,
then the making of the answer. The request only waits for it. If the request is dropped, the task goes on, so
the record is stored exactly as if the request had been answered, and the observer hears what it would have
heard. Everything before the write stays on the request's own task and is dropped with it, which is what lets
the model skip it.

**The owner's choice was to complete, not abandon** (2026-11-12). Abandoning is not on offer anyway: a write
that has been sent cannot be recalled. The choice was between finishing and reporting it, and leaving it to
chance.

**The result logs itself.** The task's output is a value that writes a `WARN` line when it is dropped without
having been taken: `a write finished after the request that asked for it had gone`, with the call, the id,
and the revision. Wherever the request disappears, before the task ends or in the instant after, the value
is dropped untaken, so there is no gap between "the request was answered" and "the log says what happened". A
write that lost its race changed nothing and says nothing. A write that failed says so and says that it may
or may not have been applied, which is the truth.

**The record's lock is held by the write too**, shared between the request and the task. Otherwise a dropped
request would let the record's next request in while the write was still on its way: it would lose the race
and be judged again, which is correct and wasteful, and an observer could hear of the two out of order.

**A retry is the caller's whole recovery.** The same request sent again is judged against the record it
stored (accepted, nothing to write) or against the old one (judged as it would have been). D35 is what makes
that safe.

**Stopping waits for them** (`Engine::settled`): a write whose request has gone belongs to no connection, so a
drain that only waited for connections would end the process with it half way. It is inside the same
deadline, and a second signal still ends the wait (D37).

**Status codes.** `DEADLINE_EXCEEDED` is this server saying early that a deadline will not be met. A deadline
that actually runs out is answered by tonic as `CANCELLED`, on the server and in a tonic client alike; other
gRPC clients report their own deadline as `DEADLINE_EXCEEDED` themselves. The load test counts both as one
thing.

**Checked over a real connection** (`lib.rs`): a caller whose deadline runs out while its write is held on its
way gets its error, the write lands, the line is logged, the retry is answered "accepted, not stored", and a
server asked to stop waits for the write.

**Not done:** a limit on writes in flight (above); an open-loop load test (D26 says why the closed one is not
one); a count of abandoned and refused requests where an operator can see it, which is a metric (proposal 26);
deadlines on the store's own calls beyond `qdrant.timeout_ms`.

## D39. A few metrics about the process, as OpenTelemetry, and nothing about a record in them

The numbers existed and went nowhere. The embedding thread counted its passes, its busy time, and its
refusals, and only the load test read them. In service the one signal was a log line per request. Nobody could
be paged on "the model is at its limit" without parsing logs.

**Three versions in one day (2026-11-12), and the third is the decision.** The first was written by hand:
atomics, a histogram, and the Prometheus text format. It worked, and the owner's judgment was that it was also
five hundred lines somebody has to maintain. The second was the `metrics` crate with its Prometheus exporter,
and, when the owner asked for a push to an OpenTelemetry collector, a bridge crate between `metrics` and the
OpenTelemetry SDK. The bridge was one person's, four hundred lines, and two releases behind the SDK, which held
every OpenTelemetry crate here back with it; the owner asked why it was needed. It was needed only because the
counting was not OpenTelemetry's to begin with. The third version records into OpenTelemetry's own metrics
API, and the `metrics` crate, its exporter, and the bridge are gone. The owner's company runs Datadog today
and is moving toward OpenTelemetry and Prometheus, and this is where it is going.

**Six, where there were twenty-one.** The owner's view, once the first two versions had counted everything
that could be counted, was that a request's trace is worth more than "how many X happened" (D40). What stays
is what no single request's trace shows:

- `record_similarity_embed_pass_duration_seconds`: how long forward passes take. The rate of its sum is the
  share of the time the model is busy, which is the server's capacity (D38).
- `record_similarity_embed_waiting`: requests waiting for the model now.
- `record_similarity_embed_refused_total{reason}` and `record_similarity_embed_abandoned_total`: callers turned
  away, and callers who gave up (D38).
- `record_similarity_writes_unanswered_total`: writes that ended after their caller had gone (D38), counted
  where their log line is written, so the two cannot disagree.
- `record_similarity_evaluations_total{verdict,mode}`: the verdicts going out. A verdict is counted with the
  write mode the request asked for, because a rejection of a dry run is a preview and a rejection under
  `enforce` is a refusal, and a rate of "rejected" that mixes them means nothing.

Gone, because a trace says it per request and a tracing backend adds it up: requests and their latency by call
and status, the store's latency and errors, the wait for a lock, the wait for the model, texts per pass,
imports, matches, readiness (the health service answers that), and the two `_info` series.

**One small file, made to be added to.** `metrics.rs` is a struct of instruments behind methods (`passed`,
`refused`, `abandoned`, `write_unanswered`, `evaluated`, `watch_waiting`). Adding a metric is a name, a field,
a line where they are made, and a method. The rest of the server calls a method, so an attribute value is a
word fixed in that file: a call site cannot invent one or spell one wrong, and "nothing about a record is an
attribute" is a property of the types and not a habit. `Metrics` is made from a meter and does not know where
its numbers go; two servers in one process, which is what the tests are, each have their own.

**Nothing about a record is an attribute.** No id, no brand, no text, no score. A brand as an attribute would
be a time series per brand per metric, in a system billed by the series, about something a monitoring system
has no business knowing. Scores are a different question with a different answer: that is proposal 10, one
line per request for judging thresholds, and it is still deferred.

**Sent to a collector, and served as a page, by the SDK.** With `telemetry.otlp_endpoint` the SDK's periodic
reader sends them over gRPC every `telemetry.metrics_interval_ms`, through its OTLP exporter, whose client is
the `tonic` already in the build. With `telemetry.metrics_listen` the SDK's Prometheus exporter keeps them for
a page, and one `axum` route serves it; `axum` was in the tree through `tonic`. The page exists because
Prometheus, a Collector's Prometheus receiver, and a Datadog Agent all read it, and a deployment that has no
collector yet can still be scraped. The names are written whole in `metrics.rs` (`_total`, `_seconds`), and
the exporter is told to add nothing, so the two routes show the same names. With neither set, the instruments
are no-ops and nothing is kept.

**Every count is there from the start, at zero**, so that a rate over it exists from the first reading and a
first occurrence is an increase, not a new series. A fresh server gave 17 series, and 31 from its first
forward pass on, however it is used.

**The buckets** run from 1 ms to 2.5 s (1, 2.5, 5, 10, 25, 50, 100, 250, 500 ms, 1 s, 2.5 s). They are a
resolution, not a threshold: nothing is decided by them. One short text takes about a millisecond and a full
pass ten to twenty.

**The names are an interface**, as the event names are (D17): add a metric or an attribute value, never
rename one or change what it counts. The second version's names that are gone were in service for a day and
on nobody's dashboard.

**A collector that is away costs only the telemetry.** The SDK retries and gives up, logs one `WARN` and one
`ERROR` for each send that failed, and sends the totals whole when the collector is back. Measured: a load run
was as fast without the collector as with it, and the server took 0.92 seconds to stop instead of 0.02.

**Checked against the real thing** (`just observe-up`: an OpenTelemetry Collector 0.130.0, Jaeger 2.8.0, and
Prometheus 3.5.0 in Docker), with Prometheus scraping the page and scraping the Collector that the server
sent to: both targets up, the same 31 series by each route, and the same values. In the tests, a collector
that keeps what it is sent (`fake-opentelemetry-collector`) receives the six from a served request.

**Not done:** a Datadog Agent, by either route; delta temporality, which some backends prefer and the
exporter can be told to use; TLS to the collector; process metrics (memory, file descriptors), which a
platform reports itself; TLS or authentication on the page, which like the gRPC port has neither and belongs
behind the mesh; and anything per brand, on purpose.

## D40. A request is a trace, and what it waited for is in it

A log line said how long a request took. Nothing said what it was waiting for, and at 64 callers a request is
25 ms of which the server's own work is 0.02. Where the rest goes decides what is worth changing, and guessing
it was wrong twice in one day: the store turned out to be the limit against Qdrant (D38), and a lock turned
out to be waited for by requests for different records (below).

**Each call is a span, and each wait is a span inside it.** The transport's layer
(`tonic-tracing-opentelemetry`) makes a span for every call, named for it, with OpenTelemetry's attributes for
a gRPC call and its status, and continues the caller's trace when the request carries a W3C `traceparent`.
Inside it the handler makes one for its own work (`evaluate`, `find_similar`, `match`, `delete`, `import`),
and inside that are the waits: `lock` for the record's lock, `store` for each call to the store with its
`operation`, and `embed` for the model, with how long the request waited for its pass (`waited_ms`) and how
many texts the pass held (`pass_texts`). The write is in the trace although it runs on a task of its own
(D38): the task is given the request's span.

**With the crates that exist for it**, by the owner's choice (2026-11-12, as D39). The first version kept a
request's waits in a task-local and put them on its log line; nothing else could read it. The second made
them `tracing` spans and a histogram each, and exported no span. The owner then said the trace was the part
worth having, and the histograms went (D39). The spans are `tracing`'s, which this workspace already logs
through; `tracing-opentelemetry` turns them into OpenTelemetry spans; and the SDK sends them in batches to
the collector named by `telemetry.otlp_endpoint`. `tonic-tracing-opentelemetry` is CC0-1.0, which the owner
allowed in `deny.toml` for it (2026-11-12).

**What a span says about a record is its id and what came of it**: the mode asked for, the verdict, whether
it was stored, its revision; a match's decision; whether a delete found anything. Never a record's text, its
brand, or a score. The id is what makes a trace findable from a complaint about a record, and it is already
on the request's log line.

**The spans are made only for the trace.** They are at `debug` and the tracing layer has a filter of its own,
so a server with no collector makes none, and the log lines do not change with or without one (D15:
`similarity-log` prints events, flat, and no spans). The layer has to be part of the one subscriber the
process has, and `tracing-subscriber` cannot add this one afterwards (its `reload` layer does not pass on
what the layer needs to find its spans), so the telemetry is started before logging is and handed to
`similarity_log::init_with` as a layer. `similarity-log` still sets up all logging and knows nothing about
OpenTelemetry.

**Health checks are not traced**: something asks every few seconds. **An `Import` stream is one trace**, with
every record's spans in it. That is right for the streams this was tried with and will be a very long trace
for a catalog; it has not been changed, because how an import should be traced is a question about how it is
used.

**Sampling is the caller's decision first.** A request that arrives inside a trace is kept or not as its
caller decided; one that starts a trace here is kept with probability `telemetry.trace_sample_ratio`, 1 by
default. Sampling saves what is sent and stored. It does not save the making of the spans.

**What it costs, measured**: 0.04 ms on an edit alone (1.61 to 1.65 ms), and at 64 callers 5% of the edits and
7% of the resaves a second, with every trace kept. With none kept it was nearly the same.

**Checked against the real thing.** `just observe-up` runs a Collector that passes traces to Jaeger. A load
run of 4 callers and 50 records left 284 traces there, one for each call, with the spans above in each. With
the ratio at 0 Jaeger was sent nothing. With the collector stopped and started again, requests were traced
from then on, and the traces of the ones it missed were lost. In the tests a request with a `traceparent` is
served over a real connection and a collector that keeps what it is sent holds six spans in the caller's
trace, each wait inside the call that made it, and none for the health check that came first.

**What the first version showed, on the development laptop** (the README has the table):

- Alone, a request is the forward pass: 1.45 of 1.48 ms for an edit. Tokenizing is 1 to 3% of a pass.
- With company it is the wait for the model: 10 of 25 ms for an edit at 64 callers, 117 of 154 for a create.
- Against Qdrant in Docker the write goes from 1.5 ms alone to 14.9 ms at 64 callers, the largest part of an
  edit there.
- The per-record lock (D10, D35) is waited for by requests for *different* records: 256 locks, picked by a
  hash of the id, so with 64 requests in flight about one in five shares a lock with another and waits out its
  whole request. 17 ms of a 154 ms create. Not changed here: this decision is the instrument, and what to do
  about each of these is the owner's to weigh. (The owner had this one changed the next day: D41.)

**Not done:** traces of `calibrate`, `import`, and the other commands, which are not requests; TLS to the
collector; trace ids on the log lines, which would join the two; and anything that samples by what a request
turned out to be (keeping every rejection, say), which is a collector's job.

## D41. A record's lock is its own

D10 gave the engine 256 locks and let a hash of the id pick one. The traces of D40 showed what that costs: with
64 requests in flight about one in five drew a lock that a request for some other record was holding, and
waited out that request's whole length, forward pass included. Nothing was protected by the wait. The two
records had nothing to do with each other.

**One lock for each record that a request is at, and none for any other** (`locks.rs`). The lock is made when
the first request for a record arrives and forgotten when the last one has left, whether that one held it or
only waited. So the table is as large as the number of requests in flight, however large the catalog is, and
there is nothing to size or tune. What each request does at the table is one lookup under a plain mutex that
is never held across a wait.

**What did not change:** requests for one record still take turns, in the order they came (the lock is still
tokio's `Mutex`); a write that outlives its request still holds the record until it has landed (D38); the wait
is still the `lock` span of a trace (D40); and across replicas there is still no lock at all, which is what
conditional writes are for (D35).

**Written here, on tokio's `Mutex`, and not taken from a crate.** The owner prefers maintained crates to
hand-written infrastructure (D39), so this was looked for first. There are crates for a lock by key
(`lockable`, `keyed-lock`, `key-mutex`), each one person's, each before 1.0, none of them what the ecosystem
uses, and the one with the most downloads last released in June 2025. What is needed beyond the mutex is a
map and a count, fifty lines, in the path that decides whether a write may happen, where this project tests
its own code hardest (D25). If the owner would rather have the dependency, `Locks::hold` is the one function
that would call it.

**Held to its rules by tests that do not need the engine.** A request never waits for a request for another
record, with 5,000 records held at once. Requests for one record are served in order. A request that gives
up waiting holds up nobody and leaves nothing in the table. And a property (D19): over any sequence of
arrivals and departures at three records, each record that anybody is at is held by exactly the first request
still there, and the table holds those records and no others. In the engine, a write that never lands for one
record leaves 2,000 other records answered and one lock in the table. That test fails on the 256 locks.

**What it bought, measured** (the README has the table). The wait is gone: over two traced load runs at 64
callers on each build, the `lock` span of a create went from a mean of 19 and 21 ms to 0.002 ms, and a fifth of
the requests had waited where now none did. The server is not faster for it, and was not expected to be once
the numbers were in: the model is the limit (D38), and a request that does not wait for a lock waits for the
model. Mean latency of a create at 64 callers was 159.6 ms and is 158.6. What changed is who waits. The 99th
percentile of a create went from 322 ms to 168, and of an edit from 57 ms to 28, because the requests that
drew a stranger's lock took twice as long as the others and now nobody does; the median of a create rose from
144 ms to 161 for the same reason. Edits a second went from 2,393 to 2,464. One caller alone, and saves with no
changes at any number of callers, were the same on both builds.

**Not done:** anything about the order in which requests for *different* records reach the model, which is the
embedding thread's (D38) and is where the remaining wait is.

## D42. The store is audited against the catalog by a command that only reads

The service's baseline for a record and the caller's own database have to say the same thing, and nothing could
check that they did. They part quietly: the caller's save fails after an `ACCEPTED`, a write is finished for a
caller that had hung up (D38), one side is restored from a backup. A stale baseline is not a cosmetic fault. The
next edit of that record is judged against something that is no longer the record.

`record-similarity audit` reads an export of the catalog, in the format an import reads (D20), lists the
collection, and reports where they disagree: a record the store lacks, one the export lacks, one whose brand,
text, or quantities differ, one stored under another layout, a point that cannot be read.

**It needs no embedding, because the store already keeps what is compared.** Beside its vectors a stored record
has the fingerprint of each field's normalized text, which is how the engine knows a field did not change, and
its quantities as numbers (D16). An export's text is fingerprinted the same way, by the one function the engine
stores them from (`PreparedRecord::hashes`, which `Plan::finish` now takes them from, held together by a
property). Text that differs only in whitespace is therefore the same text to an audit, as it is to the server,
and quantities are compared by `quantity::compare`, so `1 g` and `1000 mg` are one amount.

**What it cannot see is the anchor,** which is a composite vector with no fingerprints. Whether a record's drift
is still measured from the right original would take the model and the original text to say, and a comparison of
vectors with a tolerance somebody chose. That is not an audit, so the report says the anchor is not compared and
nothing approximates it. Storing the anchor's fingerprints would change that; it has not been needed yet.

**Listing a store is a trait of its own.** `Scan` has one method, a page of what the store holds from a cursor.
It could have been a ninth method of `VectorStore`. But the engine never scans, so the eight stores that tests
wrap around `MemoryStore` and the one that tracing wraps around any store would each have passed on a call
nobody makes; and code written against `Scan` cannot write, which is a stronger promise for a command that must
change nothing than a comment would be. Both stores implement it and pass a `scanning` half of the contract
test, and the Qdrant store has `open` beside `connect`: `connect` creates a missing collection and puts its
payload indexes back on every start, which is right for a server and wrong for a command that looks. `open`
refuses a collection that is not there and creates nothing.

**A scan lists everything, and reads no vectors.** Every point in the collection is listed whatever its layout,
each as what `get` would make of it: a record, a record of another layout, or something unreadable (a point that
still keeps its brand under `tenant`, one that does not say which record it is). Those are exactly what an
audit exists to report, so none of them ends a scan. An unreadable point names a record only if it sits where
that record is kept: anything can be written into a payload, and taking a stray point for the record it names
would report a readable record as unreadable. Vectors are left out because two composites are most of a
record's size and no part of the comparison. Feature 25 (a backup of the anchors) will need them and will have
to add them to the scan.

**A scan is not a snapshot, and the contract says what it promises anyway.** Qdrant's scroll reads each page
when it is asked for, in the order of the point ids. So: a record nobody writes during the scan is listed
exactly once; one that is rewritten is listed once, whole, as one of its versions; one created or deleted
meanwhile may be listed or not; and none is listed twice. Run against Qdrant 1.20.0, which showed what the
client's documentation leaves unsaid: a page starts at the point the last one named, a scan whose next page
starts at a point that has since been deleted goes on from the one after it, a number of points that is a whole
number of pages ends without an empty page, and a point under a number pages like one under a UUID. On a
replicated cluster a replica that lags could hide a point; the scroll's `read_consistency` is left unset, and
this was not tested, there being one node here.

**Rules that keep an audit from saying what it does not know:**

- *A record is not stored only when the whole store has been listed.* A store that cannot be listed to its end
  ends the audit with an error and no report. Every record on the page that was not read would otherwise be
  reported missing.
- *The line compared is the first one an import would have taken for that id,* because that is the line an
  import stored (it skips an id it has created). Later lines are `duplicate_id`. What a valid line is lives in
  one function, `import::prepared`, for `import --check` and the audit alike; writing it found that `--check`
  passed a blank `brand_id` that the import refuses, which is fixed.
- *A record the export names only on an invalid line is in the export.* It is not reported as missing from it;
  it is counted as not compared. The id is recovered even from a line that is refused for a key it may not
  have.
- *A record stored without a brand is not a record stored under another brand.* The first takes any request
  (D14), the second refuses them, and they are different findings.
- *An edit made since the export was taken is not a disagreement.* With `--exported-at`, a finding on a record
  written at or after that time says so and is not counted as a difference. The time is seconds since 1970,
  because every time this project keeps is, and it has no crate for dates. A record written in the very second
  the export was taken counts as written since: calling a difference that is not one costs more than setting
  one aside.
- *Everything looked at is counted once.* Each record either side has, and each line or point that names none,
  ends as exactly one of agree, differ, not stored, not in the export, or not compared. `differences` is all
  but the first, less what was written since the export, and `--fail-on-differences` fails on anything above
  zero: an audit vouches for nothing it could not compare.

Three properties (D19) hold it to what an import and the engine really do: an export that was just imported
audits clean, whatever is wrong with its lines; one change to the store or to the export is found as the kind
of change it is, and nothing else is; and the counts add up whatever the size of a page. A test under
`just test-real` does the same on the real model and a real Qdrant.

**It reports, and repairs nothing.** A record the store lacks is created by running the import on the same
export, which is all an import can do (D20). A record that differs is a question of which side is right, and
putting the caller's version in is a forced write that resets the record's anchor (D8): a decision, and the
caller's.

**The model is loaded, for its name.** The layout a server reads is the model's fingerprint and the field names
(D12), and an audit has to look for that layout: a collection whose every record is under an older model would
otherwise audit clean while every request for it failed. The fingerprint could be computed from the directory's
files without building the model, but only by a second implementation of what the loader hashes and in which
order, and a disagreement between the two would be an audit looking for the wrong layout. So the audit loads
the model exactly as `import` does, refuses what `serve` would refuse, embeds nothing, and pays for it in
memory: about 300 MB of a process that would otherwise need very little.

**The output is a report, or events** (D15, D17): text with the counts and the first few findings of each kind,
or with `--ndjson` one flat `audit_finding` for each finding, an `audit_layout` for each other layout, and an
`audit_summary` last, so that an output without a summary is known to be of an audit that did not finish. Every
count in either comes from one `Summary`. Lines carry ids and brands (the owner allowed brands in output lines
on 2026-11-11) and never a record's text or a fingerprint of it. The names and codes are an interface from
here on. An audit is not recorded as a run (D23): its arguments can hold the Qdrant API key.

**Measured,** on a laptop CPU with Qdrant in Docker on the same machine: a collection of 100,000 synthetic
records, four text fields and two quantities each, audited against its export in 1.8 seconds (three runs: 2.7,
1.8, 1.8), with the store listed at 1,000 points a page. The process peaked at 490 MB against 300 MB for the
same audit with a one-line export, so holding the export costs about 2 KB a record. Where that goes was not
measured. Each record keeps its fingerprints and its quantities in two small maps, and keeping them as arrays
in the schema's order is the first thing to try if an export ever does not fit; nothing has needed it.

**Not built:** a repair mode; an audit of anchors; an audit through the server, for a caller that cannot reach
Qdrant (the import got a streaming call for that reason, D33, and an audit could); a time given as a date.

## D43. Anchors are backed up to a file, and restored only to a record with nothing to lose

If a collection is lost, the caller's database gives back every record as it is now: an import makes a baseline
of each (D20). It cannot give back the anchors. An anchor is the first value each field of a record ever had
(D8), the server keeps vectors and never text, and the caller's database has usually forgotten what a record
said when it was created. A catalog loaded again without its anchors has every record's drift history reset to
the day of the import, and nothing says so. That is the state somebody walking a record away from what it was
would ask for.

`record-similarity anchors export` writes every record's anchor to a file, `anchors check` says whether a file
is a whole backup, and `anchors restore` puts anchors back after a collection was lost and the catalog imported
again.

**The backup is read by the scan, which now reads anchors when asked** (`Read::Anchor`, D42). It still reads
no current composite, which can be made again from the record's text. The collection is opened and never
created, the store is listed while it may be written to, and so the backup is of no single moment: each record
is in it once, with the anchor it had when its page was read. An anchor changes only when a record is created
or a write is forced, which is as much of a moment as an anchor has.

**The file is lines of JSON with a vector as bytes.** A header that names the model's fingerprint, the layout,
and the width; a line for each record with its id, its brand, its anchor, the first values of its quantities,
and the fingerprints and quantities of the record as it was; and a last line with the number of records. A
vector is its components as little-endian bytes in base64, and not a list of numbers: a backup has to give back
exactly what was stored, which bytes do by construction, and they are half the size. `base64` is the crate, at
the version `tonic` already has in the build. The last line is how a file that was cut short is known to be
one. `export` writes under another name and, when the last line is on the disk, links the file to the name that
was asked for, which fails rather than replace a file that appeared there meanwhile (a rename would not). It
never writes over a file that exists, so a file with the name that was asked for is whole.

**A backup belongs to a layout.** An anchor is a vector of one model over one set of fields (D12). A restore
reads the whole file first and refuses, having written nothing, one that is not whole or is of another layout
than the server's. After a change of model a backup is of no use, and what carries history across is still the
`anchor` an export gives a record when it is imported (D20).

**A restore gives a record an anchor only when nothing can be lost by it and nothing has to be judged.** An
anchor is the one thing an accepted write may never replace (D8), and a restore is no accepted write. So the
rule (`engine::restorable`) is the narrowest one that still does what the backup is for:

- the record is stored under the brand the backup has it under;
- it has never been written since it was created (revision 1) and is its own anchor, which is what a record is
  when it has just been imported again;
- it is, fingerprint for fingerprint and amount for amount, the record the backup was taken of.

Then it gets the anchor that was kept, extended with its own value for anything that anchor lacks, as an anchor
is for a field it never had. Every other record is left exactly as it is and named with the reason:

| Outcome | What it means |
|---|---|
| `restored` | The record has the anchor the backup kept. |
| `already` | Its anchor already is that one. Nothing was written. |
| `not_stored` | No record with that id is stored. None is created: an anchor is not a record. |
| `other_brand` | It is stored under another brand than the backup has it under. |
| `has_history` | It has an anchor that is not itself, or has been written since it was created: it was imported with its original version, edited since the import, or reset by a forced write. That history is its own. |
| `changed` | It is not what it was when the backup was taken. |

A record whose anchor a reviewer reset by a forced write since the backup is therefore never given the old one
back (it has been written), and neither is one that was imported with the original version the caller still
had (the caller's is the better source). A restore is a write like any other (D35): over the version it read
or not at all, on a task of its own, with the record's lock (D38, D41), and it counts as a write of the record.
Run twice, the second finds every anchor `already` where it should be. `--dry-run` asks the same questions and
writes nothing.

**Two vectors are one vector from a cosine of 0.999999.** Qdrant keeps a cosine vector at unit length, so an
anchor that went into it comes back as itself only to within rounding, and "is this record its own anchor" and
"is this already the anchor that was kept" are questions about vectors that have been through it. This is a
tolerance for telling a vector from itself and not a threshold on similarity: two anchors that close differ by
nothing a threshold could see, so taking one for the other changes no verdict. It was run against the real
model and Qdrant 1.20.0 (`just test-real`), where anchors that had not moved were told from the one that had.

**Held to its rule by two properties** (D19). Whatever is done to a store before a backup is taken and after,
restoring the backup writes only to a record that meets the rule, and leaves every other exactly as it was.
And a store that is lost with whatever history its records had, imported again from what the caller's database
would say, and restored, has every anchor it had. The rule and the file's reading are among the files whose
tests are tested (D25).

**For the owner to decide.** Three things here are not only technical, and were built to the most careful
answer so that nothing waits on them:

1. *Where a backup may be kept.* It holds no text. It holds ids, brands, quantities, fingerprints of text, and
   a vector made from the text each record was created with, and what a vector gives away about the text it
   was made from is not nothing. The command writes where it is told and nowhere else. The README says to give
   the file the care the collection gets; whether that is enough is a question about the owner's data and not
   about this code.
2. *Records that changed between the backup and the loss.* They are `changed` and are left with the anchor the
   import gave them, which is the record as it is now: the drift they had come before the loss is forgotten.
   How many there are depends on how old the backup is. Giving them the kept anchor would be right for an
   honest edit and wrong for an id that now holds another product, and the text cannot say which. One rule
   that could: restore where the record as it is now is still within `anchor_threshold` of the anchor that was
   kept, which is what the server would have required of every edit in between. That is a threshold deciding a
   write nobody judged, so it is not built, and the records are listed for whoever decides.
3. *A record deleted and created again with the same content* since the backup is indistinguishable from one
   that was imported again after a loss, and a restore gives it the anchor of its earlier life. Run only after
   a loss, as it is meant to be, that is the point of it. Run against a collection that was never lost, it
   would undo a delete-and-create that somebody may have meant.

**Measured,** on a laptop CPU with Qdrant in Docker on the same machine, on a collection of 100,000 synthetic
records with anchors of 1,024 components: the export took 2.5 seconds at 1,000 records a page (4.0 at 200, 2.6
at 5,000) and wrote 600 MB, about 6 KB a record; `anchors check` read that file in half a second; and a restore
that found every anchor already in place, which is one read of each record, took 20 seconds.

**Not built:** a restore for records that changed (above); a backup of records under another layout (their
vectors are another model's, and the server cannot read them either); compression of the file, which any
tool does; a backup taken through the server for a caller that cannot reach Qdrant.

## D44. One string says what a server judges by

`Describe` has always listed what a server judges by: the model, the thresholds, each field's weight and
floor, each quantity's limits. Nobody compares a dozen values across replicas, and the moment they matter is a
rolling deploy, when two requests for one record can be held to two thresholds for as long as the rollout
takes. The conditional write (D35) keeps two replicas from both writing; it does nothing about two replicas
that disagree on what to accept.

`DescribeResponse.judging_digest` is one string for all of it, and a server logs the same string on its
`serving gRPC` line, so replicas can be compared in their logs without being called.

**What is in it** is everything a verdict or a match decision depends on, and nothing else: the model's
fingerprint (two replicas with one policy and two models do not judge alike), the threshold, the drift
threshold, whether a change of measure rejects, every field's name, weight, whether it is required, and its
floor, every quantity's name and limits, and the match threshold and over-fetch. Not the address, not a batch
size, not the shadow, which decides nothing (D30).

**It is a digest of a text, and the text is pinned.** The rules are written out one to a line, in an order that
does not depend on the order the configuration lists them in, with names quoted so that no name can pass for
two, and numbers as the bits they are, so that nothing depends on how a number prints. The digest is the first
eight bytes of the SHA-256 of that text, in hexadecimal, which is how the model's own digest is written.
Servers of different versions compare digests, so the text is an interface: a test holds it to the letter, and
a new rule is a new line, never a change to one that is there.

**A rule cannot be left out by forgetting.** The function takes `Policy`, `Matching`, `FieldSpec`, and
`QuantitySpec` apart by name with nothing left over. A field added to any of them does not compile until it is
written into the text. The failure this prevents is the worst one a digest can have: two servers that judge
differently and say they agree.

**What a difference means, and does not.** The same digest means the same rules. A different one means the
configurations differ in something, and not that any verdict would: weights that are all doubled give the same
scores and another digest. Making the digest blind to that would mean deciding which differences matter, and a
digest that has opinions is one more thing to get wrong.

**The other half of the suggestion is not built:** reloading the policy without a restart. A threshold is the
setting most likely to be changed in a hurry, and today that is a restart of every replica, each loading the
model again. But a server whose rules can change while it runs is a different thing to operate than one whose
rules are what it started with, and that is the owner's to choose. If it is wanted, three things are already
clear. Only what needs no re-index may be reloaded (D12): thresholds, weights, floors, limits, and never the
model or the set of fields. A reload that does not parse, or that would change the layout, should leave the
old policy in force and say so loudly, since a server that stops on a typo turns a mistake into an outage. And
the digest is what tells whoever sent the signal that every replica took it.

## D45. ONNX Runtime's threads spin, and how many there are is a setting of the machine

Asked how to make the server faster, the first candidate was quantizing the model to 8-bit integers. It was
measured first, on a copy of the model and in Python's ONNX Runtime 1.32 on the README's laptop (an M4 Pro, 8
fast cores and 4 slow ones), and it is not worth having there: one short field went from 0.82 to 0.56 ms, while
a description (2.09 to 2.02 ms) and a batch of 32 texts (27.1 to 25.4 ms), which are most of the work, hardly
moved. Per-channel quantization kept every vector within a cosine of 0.9974 of the original and per-tensor
within 0.9838, but a different vector is a different model, with a new fingerprint, a re-index, and a new
calibration, and the load check of D34 would have to be loosened to admit it. ONNX Runtime's transformer
fusion found no attention to fuse and gained nothing. x86-64 processors with fast 8-bit instructions may say
otherwise, and are not measured.

What the same measurements did show is that the Python runtime ran a pass much faster than the server on the
same file and the same cores. The difference was spinning: a thread of ONNX Runtime's pool that finishes its
part of one step of a pass either waits busily for the next or goes to sleep and is woken. A pass of this model
is many small steps, and a sleeping thread is woken late for each. The session the server built did not ask
for spinning, and ran exactly as it does with spinning turned off. The `ort` crate's documentation says intra-op
spinning is on by default; whatever the reason, this build did not behave so, and the call that turns it on is
what made the difference measured below.

**Spinning is on, always.** Measured with `just throughput` and `just load`, same laptop, release build,
default model, the number of threads left to ONNX Runtime:

| | Before | Spinning |
|---|---|---|
| One short field | 1.31 ms | 0.78 ms |
| One whole record | 4.27 ms | 2.92 ms |
| Eight whole records | 16.3 ms | 13.2 ms |
| One edit through gRPC, one caller | 1.64 ms | 1.11 ms |
| Edits a second, 16 callers | 1,834 | 2,165 |
| Edits a second, 64 callers | 2,479 | 2,807 |
| Creates a second, 64 callers | 399 | 483 |
| One `Import` stream, records a second | 398 | 483 |

The vectors are the same: the arithmetic is unchanged, only how the threads wait for it. So nothing about it is
in the fingerprint, the layout, or the judging digest (D44), and no threshold moves. It costs CPU while a pass
runs, about a third more for the same work under load in one measurement with 6 threads (6.8 s against 5.0 s
of CPU for the same requests), and none at rest: a server that had been idle for 10 seconds had used no CPU
in them either way.
That is a cost on a machine the server shares, and there is no setting to turn it off; one can be added if a
deployment needs it.

**How many threads is `model.threads`, and is left out by default.** In a probe of the runtime alone, 8
spinning threads looked best. Through gRPC, where the server's own threads and the load test's want cores too,
8 was worse than 6, and 4 was worse again. With spinning on and the
number left to ONNX Runtime, the server was as fast as with 6 at every level of `just load` (2,807 against
2,809 edits a second at 64 callers). So `config.example.toml` sets none. The setting exists for the machines
where ONNX Runtime's own count is wrong, most likely a container that is given a share of a larger machine and
sees all of its cores. It is checked to be positive, written on the line that says the model was loaded, and
`just throughput <dir> onnx <threads>` measures a number before it is configured.

**What was measured before this is labeled so.** In the README, where a request's time goes, a server
against Qdrant, the Linux container, the `import` command, and a stream beside edits are from builds before
spinning, and say so; the per-record locks are a comparison of two builds and say which. They still show what
they were measured to show (shares of a request, the store as the limit, Linux against macOS, what a stream
costs its neighbors). The figures that size a server were measured again: the forward pass, `just load` at 1
to 64 callers, and the deadline runs at 128 and 256.

**Not done:** several embedding threads, each with a session of fewer threads (D34 lists it), which is the next
thing to measure now that a pass no longer waits on its own threads (since measured and built: D46); the Linux container and x86-64 with
spinning; and quantization on x86-64.

## D46. Several sessions of the model can serve at once, and one is the default

D45 left one idea to measure: several embedding threads, each with a session of fewer threads. One session
spreads a pass across every core it is given, and a pass is many small steps; between them, cores wait for each
other. Sessions side by side keep more cores working. It was measured first, then built, then measured through
the server.

**What the model alone does.** A probe ran k sessions of t threads each, every one embedding the same batch over
and over, on the README's laptop (M4 Pro, 8 fast cores and 4 slow ones), release build, default model. Texts a
second, all sessions together:

| Records a pass | 1 session, ONNX Runtime's count | 2 × 4 threads | 4 × 2 | 8 × 1 |
|---|---|---|---|---|
| 1 | 1,344 | 2,042 | 2,793 | 3,347 |
| 4 | 1,935 | 2,930 | 3,559 | 4,475 |
| 16 | 2,254 | 2,918 | 3,561 | 4,321 |

The price is the pass itself: one record takes 2.97 ms in a session with every thread, 3.53 ms with four, 4.43
with two, and 5.87 with one, when nothing else runs; side by side, longer.

**What was built.** `model.sessions`, one unless given, is how many sessions the server loads (each checked
against the weights as D34 requires, and each the same fingerprint), each on a thread of its own. They share one
backlog under one lock, which a thread takes only to put its next pass together, so D38 holds as it was: the
order requests are served in, the switch to the newest while requests are being lost, the bound that refuses
only what cannot be met, are one decision whoever makes it. The one rule that had to change is the last step of
putting a pass together: a request left waiting was told it was too late when the pass about to run would end
after its deadline. With other sessions, another may be free for it sooner, so it is now told only when its
deadline comes before the soonest any session can be free, at the fastest rate seen. A session that is idle
refuses nobody. With one session the rule is what it was.

`model.threads` becomes the threads of each session, and is required when `sessions` is more than one: left to
ONNX Runtime, every session would take every core. Neither setting changes a vector, so neither is in the
fingerprint, the layout id, or the judging digest (D44). The worker is alive (D37) while any session's thread is.

**Measured through the server.** `just load`, same laptop with a browser and a terminal open (which is why the
one-session figures are below the README's: everything here was run side by side under the same conditions, and
the ratios are the result, not the numbers), 1,000 records a level, two runs of each alternating:

| | 1 session | 4 × 2 | 8 × 1 |
|---|---|---|---|
| Edits a second, 1 caller | 709, 769 | 552, 570 | 493, 498 |
| Median edit, 1 caller | 1.38, 1.27 ms | 1.79, 1.73 ms | 2.01, 2.01 ms |
| Edits a second, 64 callers | 2,341, 2,526 | 3,174, 3,503 | 3,859, 3,966 |
| Median edit, 64 callers | 27.0, 25.2 ms | 16.3, 17.5 ms | 8.0, 11.2 ms |
| 99th percentile edit, 64 callers | 30.9, 33.8 ms | 51.8, 41.8 ms | 68.6, 48.8 ms |
| Creates a second, 64 callers | 405, 444 | 722, 774 | 955, 982 |
| One `Import` stream, records a second | 392 | 280 | 189 |

With a 50 ms deadline and 20,000 records (D38's run), edits answered in time a second, two runs:

| Callers | 1 session | 4 × 2 | 8 × 1 | 6 × 1 | 3 × 2 |
|---|---|---|---|---|---|
| 128 | 2,196 | 4,051, 3,740 | 4,207, 3,750 | 3,676 | 2,955 |
| 256 | 2,244 | 3,516, 2,987 | 1,666, 1,085 | 2,476 | 2,870 |

So: under load, sessions do much more work on the same cores: 1.6 times the edits and 2.2 to 2.4 times the
creates with 8 × 1, and at the median a loaded server answers sooner. Everything else costs. A request alone is
about a third slower with 4 × 2 and up to 60% with 8 × 1, the slowest requests under load are slower, and an `Import` stream, which sends one batch at a
time and so only ever uses one session, is slower in proportion to the threads its session lost. With the most
sessions, a server well past its capacity with deadlines answers fewer in time than one session does: at 256
callers 8 × 1 fell to half. Most of those were lost after the model had answered them in time. A rule that
treated a request answered after its deadline as lost (so that the next pass would take the newest) was tried
and changed nothing, which says the time went after the model: the sessions held every fast core, and gRPC, the
runtime, and the load test's own callers waited for one. 4 × 2 leaves room for them: it was the best of those tried at 256
callers, and close to 8 × 1 at 128.

**One is the default, and `config.example.toml` sets none.** Which is better depends on what a deployment
needs: a quiet server answers a single request faster with one session, a busy one serves more with several,
and a server that imports by stream wants one. That is a choice for the machine and the traffic, like
`threads`, and it is measured with `just load` on the machine that will run it. Leave cores for the rest of the
server: sessions times threads at or below the fast cores, and fewer when it is past its capacity.

**Not done.** A long job split across idle sessions, which would give a stream its speed back. The `import`
command embeds directly and uses one session of `threads` threads, so with `threads` set for several sessions it
is slower too; it could be given a count of its own. Linux, the container, and x86-64 are not measured, and
nothing here was run against Qdrant, which D38 found to be the limit before the model is.

## D47. Quantities are also a service of their own, which computes nothing the engine does not

The owner asked for the quantity comparison to be callable without a record: other systems need to know whether
two amounts are the same, and weights are intricate enough (an eighth of an ounce is not 3.5 g) to deserve tests
that state what the service makes of them. So the package has a second service, `Quantities`, with three calls:
`Compare` (two versions' quantities, name by name, answered with the `QuantityScore` of D16), `Convert` (one
quantity in another unit, or in the unit its dimension is compared in), and `ListUnits` (the table).

**It is the engine's code, not a copy of it.** `Compare` is `quantity::compare`, called the way `Evaluate` calls
it. `Convert` is `Quantity::in_unit`, which is new, and is the conversion `compare` already made internally: a
property holds that a quantity converted to another unit of its dimension is `Unchanged` against itself, and that
the conversion is the arithmetic the properties of D16 were already checking by hand. `ListUnits` reads the one
table. There is no second place where a unit or a factor is written, so the service cannot say something about
two amounts that a verdict would contradict.

**A service of its own, and not three more calls on `RecordSimilarity`.** That service is about records: its
readiness is the store and the model, its calls honor deadlines against the embedding threads (D38), and its
names are held to the configured schema. None of that is true of a conversion. As a separate service it has a
health name of its own (`recordsimilarity.v1.Quantities`), which is `SERVING` from the start until the server
leaves the rotation to stop (D37) and does not go away when Qdrant does, and a client that wants only this
generates only this.

**It reads nothing of the server.** No schema: a name is whatever the caller says, since a quantity that this
deployment does not define is exactly what another system will ask about. No limits: `beyond_limit` is never
set, because a limit is a measurement that belongs to a schema (D16), and applying this server's to another
system's amounts would be a decision made for it. No store, no model, no lock. The same request gets the same
answer from any server of this version, whatever its configuration, which is what makes the answers fit to be
written into a test.

**Units that do not convert are an answer.** `Convert` from `%` to `mg` returns no quantity and both
dimensions, for the reason `Compare` reports `INCOMPARABLE` and does not fail: it is a fact about the two units.
A unit the table does not know stays its own dimension, comparable with itself (D16). What is refused,
`INVALID_ARGUMENT`, is what a record is refused for (an amount that is not a usable measurement), a missing
quantity, a blank target unit, and an amount that is a number in its own unit and not in the one asked for
(`1e300 kg` in `mcg`). Another spelling of the same unit changes no number: `v * f / f` is not always `v`, and
"oz" to "ounces" should not be arithmetic.

**What it adds to operations is small on purpose.** A span for each of the two calls that compute
(`compare_quantities` with how many, `convert_quantity` with whether there was one, at `debug` like the others,
D40), one log line each with those and `elapsed_ms`, and no metric (D39: a count of calls is what a trace is
for). Nothing of a request's names, units, or amounts is in either.

**Not done.** Parsing an amount out of text (`"3.5g"`, `"1/8 oz"`): the service takes a number and a unit, as a
record does. Units per something (`mg/ml`, `mg` per serving against `mg` per package), which are different
dimensions to the table and so are comparable only with themselves. A limit on how many quantities one `Compare`
may carry beyond the transport's message size. A Bruno request for the three calls. `just mutants` has not been
run over the new code in `quantity.rs`; it is covered by example tests written to its boundaries and by the
property above, and the run is owed (D25).
