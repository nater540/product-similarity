# models/

Model files are fetched, not vendored. Everything in this directory except this note is ignored by git.

```sh
just fetch-model        # or: cargo run --release -p similarity-server -- model fetch
```

That downloads the default model into `models/mdbr-leaf-mt/`:

| | |
|---|---|
| Model | [`MongoDB/mdbr-leaf-mt`](https://huggingface.co/MongoDB/mdbr-leaf-mt) |
| Revision | `1ed41b22ce166d66c24f88ebfc340e1f03adb20f` |
| License | Apache-2.0 |
| Size | 23M parameters, about 92 MB on disk |
| Architecture | 6-layer BERT (384 hidden), mean pooling, linear projection to 1024 dimensions |

The revision is pinned and the SHA-256 of every file is compiled into the binary
(`crates/similarity-server/src/fetch.rs`), so a fetch either produces exactly the bytes the parity fixture and the
calibration were produced with, or fails.

Any BERT-architecture sentence-transformers model can be fetched the same way and used by pointing `model.dir` at
it, for example:

```sh
record-similarity model fetch --repo sentence-transformers/all-MiniLM-L6-v2 \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --dir models/all-MiniLM-L6-v2
```

No checksums are pinned for other models; the command prints the digests of what it downloaded. A different model
has a different fingerprint, so it needs its own Qdrant collection and its own calibration: thresholds do not
transfer between models (see `docs/decisions.md`, D3). Remove `output_dim` from the config for any model that was
not trained to be truncated (D4).

Fine-tuned models written by `record-similarity-train` (`just finetune`) belong here too; see "Fine-tuning the
model" in the top-level README.

Every directory these tools write holds a `provenance.json`: for a fetched model, the repository, the revision,
and the checksum of each file; for a trained one, the base model, the data, the options, and the base's own
provenance nested inside. It is a note about the model and not part of it, so it changes no fingerprint. See
"Where a model came from" in the top-level README.
