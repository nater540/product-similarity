# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers>=5.1,<6", "torch>=2.6"]
# ///
"""Generates the PyTorch parity fixture that `crates/similarity-core/tests/real_parity.rs` checks the Burn port against.

Run from the repository root after `record-similarity model fetch`:

  uv run tools/golden.py models/mdbr-leaf-mt crates/similarity-core/tests/fixtures/golden_mdbr-leaf-mt.json

The fixture stores the reference pipeline's own L2-normalized output, so the Rust test compares like with like. It is
checked in: regenerating it needs PyTorch, running the test does not.
"""

import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

MAX_TOKENS = 256

# Short categorical values, typical names, a long description that must truncate, and text that exercises the
# tokenizer's normalizer (case, accents, punctuation, non-Latin script).
TEXTS = [
  "Blueberry Gummies 100mg",
  "Blue Dream Vape Cartridge 1g",
  "Edibles > Gummies",
  "Concentrates > Live Resin",
  "WYLD",
  "Ten vegan gummies infused with hybrid enhanced botanical terpenes. 10mg THC per piece, 100mg per package. "
  "Made with real fruit and natural flavoring.",
  "Crème Brûlée PRE-ROLL (0.5g) — Indica; 24.7% THC!!",
  "ブルーベリー グミ 100mg",
  "a",
  " ".join(["Slow-cured, hand-trimmed indoor flower with dense purple buds and a sweet berry aroma."] * 40),
]


def main() -> None:
  model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
  model = SentenceTransformer(str(model_dir), device="cpu")
  model.max_seq_length = MAX_TOKENS
  embeddings = model.encode(TEXTS, batch_size=len(TEXTS), normalize_embeddings=True, convert_to_numpy=True)
  # One text encoded alone: padding must not change the result, so this catches attention-mask bugs in the port.
  alone = model.encode([TEXTS[0]], normalize_embeddings=True, convert_to_numpy=True)
  assert abs(float((alone[0] * embeddings[0]).sum()) - 1.0) < 1e-5

  fixture = {
    "model": model_dir.name,
    "max_tokens": MAX_TOKENS,
    "dimension": int(embeddings.shape[1]),
    "texts": TEXTS,
    "embeddings": [[round(float(value), 7) for value in row] for row in embeddings],
  }
  out_path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
  print(f"wrote {len(TEXTS)} embeddings of dimension {embeddings.shape[1]} to {out_path}")


if __name__ == "__main__":
  main()
