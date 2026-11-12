# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers>=5.1,<6", "torch>=2.6"]
# ///
"""Generates the PyTorch reference that `crates/similarity-train/tests/real_parity.rs` checks the Burn trainer against.

Run from the repository root after `record-similarity model fetch`:

  uv run tools/train_golden.py models/mdbr-leaf-mt crates/similarity-train/tests/fixtures/training_mdbr-leaf-mt.json

It takes a few optimizer steps on fixed batches, with dropout off so the run is deterministic, and records what a
correct trainer must reproduce: the loss and the global gradient norm of every step (the forward pass, both losses,
and the whole backward pass), and the embeddings of some probe texts afterwards (the optimizer and the clipping).

The fixture is checked in: regenerating it needs PyTorch, running the test does not. This script is a referee, not
part of the training pipeline, which is `record-similarity-train` (Rust, Burn).
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

MAX_TOKENS = 256
DIMS = [1024, 512, 256]
MARGIN = 0.5
SCALE = 20.0
# Larger than a real run's 2e-5, so that a handful of steps move the embeddings far enough to tell a right update
# from a wrong one.
LEARNING_RATE = 5e-5
# Lower than a real run's 1.0, so that some of these steps are clipped and some are not.
MAX_GRAD_NORM = 0.3

# (before, after, is the same product). The recycles are deliberately near misses: products that share most of
# their words, which is where the loss and its gradient are not zero.
PAIRS = [
  ("Blueberry Gummies 100mg", "Blueberry Gummies 100mg (10 pack)", True),
  ("Blue Dream Vape Cartridge 1g", "Blue Dream 1g Cart - Sativa", True),
  ("Blueberry Gummies 100mg", "Strawberry Gummies 100mg", False),
  ("Blue Dream Vape Cartridge 1g", "Blue Dream Pre-Roll 1g", False),
  ("OG Kush Pre-Roll 1g", "OG Kush Pre Roll (1 gram)", True),
  ("Live Resin Badder 1g", "Live Rosin Badder 1g - Sour Diesel", False),
  ("Ten vegan gummies infused with hybrid terpenes. 10mg THC per piece.", "Ten vegan gummies. 10mg THC each, 100mg total.", True),
  ("Ten vegan gummies infused with hybrid terpenes. 10mg THC per piece.", "Ten mints infused with hybrid terpenes. 5mg THC per piece.", False),
]

# (catalog entry, a retailer's menu item for it). Several flavors of one product line, so that telling them apart
# takes more than the brand name.
GROUPS = [
  ("WYLD Raspberry Sativa Gummies 100mg", "Wyld - Raspberry Gummies (Sativa) 10pk"),
  ("WYLD Marionberry Indica Gummies 100mg", "wyld marionberry gummies indica 10 pack"),
  ("WYLD Huckleberry Hybrid Gummies 100mg", "Huckleberry hybrid gummies by WYLD"),
  ("WYLD Elderberry Indica Gummies 2:1 THC:CBN", "Wyld elderberry sleep gummies with CBN"),
  ("Kiva Camino Pineapple Habanero Gummies", "CAMINO pineapple habanero 100mg gummies by Kiva"),
  ("Kiva Camino Midnight Blueberry Gummies", "Camino midnight blueberry sleep gummies"),
  ("STIIIZY OG Kush Pod 1g", "Stiiizy pod - OG Kush - 1 gram"),
  ("STIIIZY Blue Dream Pod 1g", "STIIIZY blue dream 1g pod (sativa)"),
]

PROBES = ["Blueberry Gummies 100mg", "Blue Dream Vape Cartridge 1g", "Wyld raspberry gummies", "a"]

# Which batch each step trains on.
STEPS = ["pairs", "groups", "pairs", "groups", "pairs", "groups"]


def unit_prefix(embeddings, dim):
  return F.normalize(embeddings[:, :dim], dim=-1)


def contrastive(a, b, same):
  distance = 1.0 - (a * b).sum(dim=-1)
  return 0.5 * (same * distance.pow(2) + (1.0 - same) * F.relu(MARGIN - distance).pow(2)).mean()


def ranking(a, b):
  scores = a @ b.T * SCALE
  return F.cross_entropy(scores, torch.arange(len(scores)))


def main() -> None:
  model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
  torch.manual_seed(0)
  model = SentenceTransformer(str(model_dir), device="cpu")
  model.max_seq_length = MAX_TOKENS
  model.eval()  # dropout off; gradients still flow

  # sentence-transformers renamed `tokenize` to `preprocess` during the 5.x series.
  prepare = getattr(model, "preprocess", None) or model.tokenize

  def encode(texts):
    return model(prepare(list(texts)))["sentence_embedding"]

  with torch.no_grad():
    native = encode(["a"]).shape[1]
  dims = [dim for dim in DIMS if dim <= native]

  def probes():
    with torch.no_grad():
      return [[round(float(value), 7) for value in row] for row in encode(PROBES)]

  parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
  optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

  before = probes()
  steps = []
  for kind in STEPS:
    optimizer.zero_grad()
    if kind == "pairs":
      left, right = encode(pair[0] for pair in PAIRS), encode(pair[1] for pair in PAIRS)
      same = torch.tensor([1.0 if pair[2] else 0.0 for pair in PAIRS])
      losses = [contrastive(unit_prefix(left, dim), unit_prefix(right, dim), same) for dim in dims]
    else:
      left, right = encode(group[0] for group in GROUPS), encode(group[1] for group in GROUPS)
      losses = [ranking(unit_prefix(left, dim), unit_prefix(right, dim)) for dim in dims]
    loss = sum(losses) / len(losses)
    loss.backward()
    # Returns the norm before clipping, which is what the Rust trainer reports too.
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
    optimizer.step()
    steps.append({"kind": kind, "loss": loss.item(), "grad_norm": grad_norm.item()})
    print(f"{kind:6}  loss {loss.item():.6f}  grad norm {grad_norm.item():.6f}")

  fixture = {
    "model": model_dir.name,
    "max_tokens": MAX_TOKENS,
    "dims": dims,
    "margin": MARGIN,
    "scale": SCALE,
    "learning_rate": LEARNING_RATE,
    "max_grad_norm": MAX_GRAD_NORM,
    "pairs": [{"before": before_, "after": after, "same": same} for before_, after, same in PAIRS],
    "groups": [{"catalog": catalog, "menu_item": menu_item} for catalog, menu_item in GROUPS],
    "steps": steps,
    "probes": PROBES,
    "probes_before": before,
    "probes_after": probes(),
  }
  out_path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
  print(f"wrote {len(steps)} reference steps to {out_path}")


if __name__ == "__main__":
  main()
