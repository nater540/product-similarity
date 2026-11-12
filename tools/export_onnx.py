# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers>=5.1,<6", "torch>=2.6", "onnx>=1.17", "onnxruntime>=1.20"]
# ///
"""Writes `onnx/model.onnx` into a sentence-transformers model directory, for the server's ONNX Runtime engine.

Run from the repository root:

  uv run tools/export_onnx.py models/mdbr-leaf-mt

This is in Python because the exporter that turns a PyTorch module into an ONNX graph is PyTorch's own, and there
is nothing to gain from writing a second one. It is a converter, run once for each model directory (again after
fine-tuning: the weights are baked into the file). Nothing at run time depends on Python.

The graph is the directory's pipeline up to, and not including, normalization: tokens and their mask in, one
vector of the model's native width per text out. That is exactly what the Burn encoder returns, so the server
truncates and normalizes both the same way. The server does not take this file on trust: when it loads the ONNX
engine it also loads the directory's weights in Burn and requires the two to agree on a set of texts.
"""

import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize

OPSET = 17
# Checked after the export: short and long, so that padding, the mask, and pooling are all exercised.
TEXTS = [
  "Blueberry Gummies 100mg",
  "Edibles > Gummies",
  "Ten vegan gummies infused with hybrid enhanced botanical terpenes. 10mg THC per piece, 100mg per package.",
  "WYLD",
]


class Pipeline(torch.nn.Module):
  """The directory's modules in order, without a final normalization."""

  def __init__(self, model: SentenceTransformer):
    super().__init__()
    self.stages = torch.nn.ModuleList(stage for stage in model if not isinstance(stage, Normalize))

  def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    features = {"input_ids": input_ids, "attention_mask": attention_mask}
    for stage in self.stages:
      features = stage(features)
    return features["sentence_embedding"]


def main() -> None:
  if len(sys.argv) != 2:
    sys.exit("usage: export_onnx.py <model-dir>")
  directory = Path(sys.argv[1])
  model = SentenceTransformer(str(directory), device="cpu")
  pipeline = Pipeline(model).eval()
  encoded = model.tokenizer(TEXTS, padding=True, truncation=True, return_tensors="pt")
  inputs = (encoded["input_ids"], encoded["attention_mask"])

  target = directory / "onnx" / "model.onnx"
  target.parent.mkdir(exist_ok=True)
  # Written beside the target and renamed, so that a server never reads half a file.
  partial = target.with_suffix(".onnx.partial")
  torch.onnx.export(
    pipeline,
    inputs,
    str(partial),
    input_names=["input_ids", "attention_mask"],
    output_names=["embedding"],
    dynamic_axes={
      "input_ids": {0: "batch", 1: "tokens"},
      "attention_mask": {0: "batch", 1: "tokens"},
      "embedding": {0: "batch"},
    },
    opset_version=OPSET,
    dynamo=False,
  )
  onnx.checker.check_model(str(partial))

  # The file has to compute what PyTorch computes, on a batch of another shape than the one it was traced with.
  session = onnxruntime.InferenceSession(str(partial), providers=["CPUExecutionProvider"])
  worst = 0.0
  for texts in (TEXTS, TEXTS[:1], TEXTS[2:3] * 3):
    batch = model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
      expected = pipeline(batch["input_ids"], batch["attention_mask"]).numpy()
    found = session.run(None, {"input_ids": batch["input_ids"].numpy(), "attention_mask": batch["attention_mask"].numpy()})[0]
    worst = max(worst, float(np.abs(expected - found).max()))
  if worst > 1e-4:
    partial.unlink()
    sys.exit(f"the exported graph differs from PyTorch by {worst:.2e}; nothing was written")
  os.replace(partial, target)
  width = session.get_outputs()[0].shape[-1]
  print(f"wrote {target} ({target.stat().st_size / 1e6:.0f} MB, width {width}, largest difference from PyTorch {worst:.1e})")


if __name__ == "__main__":
  main()
