# /// script
# requires-python = ">=3.10"
# dependencies = ["grpcio>=1.66", "grpcio-tools>=1.66"]
# ///
"""A walk through the service for an audience, against a running server.

It makes a synthetic catalog (every brand and product here is invented, and the same seed makes the same catalog),
loads it over the `Import` stream, and then sends real requests about it: an ordinary edit, a product recycled for
another product, a product moved toward another in small steps, a quantity that changes beside unchanged text, a
request from the wrong brand, a retailer's menu item matched to the catalog, and a neighbor search. Each step prints
what was sent, what the server answered, and how long the call took. Nothing is asserted: what is printed is what
came back.

  just serve-in-memory            # in one terminal: a server that keeps nothing
  just demo --pause               # in another: one step per Enter

Why this is Python and not Rust: it is a client, and a script for one occasion. It shows the API being called from
another language with stubs generated from the `.proto` at start, which is how any caller that is not Rust would do
it. Nothing the server builds, runs, or tests needs it.

It writes to the server it is pointed at (`demo-NNNN` ids, created and then deleted again on the next run), so it
refuses an address that is not this machine unless `--allow-remote` is given. Point it at a server made for it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import statistics
import sys
import tempfile
import textwrap
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import grpc
from grpc_tools import protoc

PROTO_ROOT = Path(__file__).resolve().parent.parent / "crates" / "similarity-proto" / "proto"
PROTO_FILE = "recordsimilarity/v1/similarity.proto"

# --- the catalog ---------------------------------------------------------------------------------------------------

# Invented brands. Each makes a few kinds of product, so that each one has a catalog of its own.
BRANDS = [
  ("Sunny Orchard", ["gummies", "chocolate", "seltzer"]),
  ("Tidewater Extracts", ["cartridge", "concentrate"]),
  ("Northwind Gardens", ["flower", "preroll"]),
  ("Copper Kettle Confections", ["chocolate", "gummies"]),
  ("Lantern Pine", ["tincture", "topical"]),
  ("Slow River Farms", ["flower", "preroll", "concentrate"]),
  ("Paper Crane", ["cartridge", "gummies"]),
  ("Marigold Lane", ["topical", "tincture", "seltzer"]),
  ("Foxglove Farms", ["flower", "cartridge"]),
  ("Quiet Harbor", ["seltzer", "gummies", "tincture"]),
  ("Blue Heron Botanicals", ["concentrate", "preroll", "topical"]),
  ("Desert Bloom Supply", ["flower", "chocolate", "cartridge"]),
]

STRAINS = {
  "Blue Dream": ("sativa", "blueberry and sweet herbs"),
  "Sour Diesel": ("sativa", "citrus and fuel"),
  "Granddaddy Purple": ("indica", "grape and berry"),
  "Northern Lights": ("indica", "pine and sweet spice"),
  "Jack Herer": ("sativa", "pine and pepper"),
  "Wedding Cake": ("hybrid", "vanilla and earthy pepper"),
  "Gelato": ("hybrid", "sweet cream and citrus"),
  "Pineapple Express": ("hybrid", "pineapple and cedar"),
  "Bubba Kush": ("indica", "coffee and cocoa"),
  "Durban Poison": ("sativa", "anise and sweet lemon"),
  "Zkittlez": ("indica", "tropical fruit and candy"),
  "Green Crack": ("sativa", "mango and citrus"),
  "Purple Punch": ("indica", "grape candy and vanilla"),
  "Super Lemon Haze": ("sativa", "lemon peel and candy"),
  "White Widow": ("hybrid", "earth and sweet wood"),
  "Strawberry Cough": ("sativa", "strawberry and fresh herbs"),
  "Mimosa": ("hybrid", "orange and tropical fruit"),
  "Do-Si-Dos": ("indica", "mint and sweet earth"),
}

FLAVORS = [
  "Blueberry", "Watermelon", "Sour Cherry", "Peach Mango", "Strawberry Lemonade", "Green Apple", "Pineapple",
  "Blood Orange", "Raspberry", "Concord Grape", "Tangerine", "Passionfruit",
]


def amount(value: float) -> str:
  """A number as a label would print it: `10`, `2.5`."""
  return f"{value:g}"


def gummies(rng: random.Random) -> tuple[dict, dict]:
  flavor, total, count = rng.choice(FLAVORS), rng.choice([50, 100, 200]), rng.choice([10, 20])
  lean, diet = rng.choice(["indica", "sativa", "hybrid"]), rng.choice(["vegan", "gluten-free", "pectin-based"])
  return (
    {
      "name": f"{flavor} Gummies {total}mg",
      "category": "Edibles > Gummies",
      "description": f"{count} {diet} fruit gummies, {lean}-leaning, with a ripe {flavor.lower()} flavor. "
      f"{amount(total / count)}mg THC per piece, {total}mg per pouch.",
    },
    {"weight_per_item": {"value": count * 4, "unit": "g"}, "thc_per_item": {"value": total, "unit": "mg"}},
  )


def chocolate(rng: random.Random) -> tuple[dict, dict]:
  kind, total = rng.choice(["Dark", "Milk", "White"]), rng.choice([100, 200])
  extra = rng.choice(["Sea Salt", "Espresso", "Mint", "Almond", "Caramel", "Raspberry", "Hazelnut", "Orange Peel"])
  squares = rng.choice([10, 20])
  return (
    {
      "name": f"{extra} {kind} Chocolate Bar {total}mg",
      "category": "Edibles > Chocolates",
      "description": f"A {kind.lower()} chocolate bar with {extra.lower()}, scored into {squares} squares of "
      f"{amount(total / squares)}mg THC each. {total}mg THC per bar.",
    },
    {"weight_per_item": {"value": 50, "unit": "g"}, "thc_per_item": {"value": total, "unit": "mg"}},
  )


def seltzer(rng: random.Random) -> tuple[dict, dict]:
  flavor = rng.choice(["Lime", "Grapefruit", "Mango", "Black Cherry", "Cucumber Mint", "Yuzu", "Blackberry", "Ginger"])
  total = rng.choice([5, 10, 25])
  return (
    {
      "name": f"{flavor} Seltzer {total}mg",
      "category": "Beverages > Seltzers",
      "description": f"A lightly carbonated {flavor.lower()} seltzer with no added sugar. {total}mg THC in a "
      "12 fl oz can, made to come on in about fifteen minutes.",
    },
    {"volume_per_item": {"value": 355, "unit": "ml"}, "thc_per_item": {"value": total, "unit": "mg"}},
  )


def cartridge(rng: random.Random) -> tuple[dict, dict]:
  strain, size, oil = rng.choice(list(STRAINS)), rng.choice([0.5, 1]), rng.choice(["Distillate", "Live Resin"])
  lean, notes = STRAINS[strain]
  low = rng.randrange(78, 90)
  return (
    {
      "name": f"{strain} {oil} Cartridge {amount(size)}g",
      "category": "Vape Pens > Cartridges",
      "description": f"A {amount(size)} gram 510-thread cartridge of {strain} {oil.lower()}, a {lean} with notes "
      f"of {notes}. No cutting agents.",
    },
    {"weight_per_item": {"value": size, "unit": "g"}, "thc_per_item": {"min": low, "max": low + 4, "unit": "%"}},
  )


def flower(rng: random.Random) -> tuple[dict, dict]:
  strain, weight = rng.choice(list(STRAINS)), rng.choice([3.5, 7, 14, 28])
  lean, notes = STRAINS[strain]
  low = rng.randrange(18, 28)
  return (
    {
      "name": f"{strain} {amount(weight)}g",
      "category": f"Flower > {lean.capitalize()}",
      "description": f"Indoor-grown {strain}, a {lean} with dense, frosty buds and notes of {notes}. Hand trimmed "
      "and slow cured.",
    },
    {"weight_per_item": {"value": weight, "unit": "g"}, "thc_per_item": {"min": low, "max": low + 3, "unit": "%"}},
  )


def preroll(rng: random.Random) -> tuple[dict, dict]:
  strain, pack = rng.choice(list(STRAINS)), rng.choice([1, 5])
  lean, notes = STRAINS[strain]
  each = 1 if pack == 1 else 0.5
  low = rng.randrange(18, 28)
  name = f"{strain} Pre-Roll 1g" if pack == 1 else f"{strain} Pre-Rolls 5-Pack 2.5g"
  return (
    {
      "name": name,
      "category": "Pre-Rolls > Singles" if pack == 1 else "Pre-Rolls > Packs",
      "description": f"{'One' if pack == 1 else 'Five'} {amount(each)} gram pre-roll{'' if pack == 1 else 's'} of "
      f"whole-flower {strain}, a {lean} with notes of {notes}. No trim, no shake.",
    },
    {"weight_per_item": {"value": pack * each, "unit": "g"}, "thc_per_item": {"min": low, "max": low + 3, "unit": "%"}},
  )


def concentrate(rng: random.Random) -> tuple[dict, dict]:
  strain, form = rng.choice(list(STRAINS)), rng.choice(["Live Resin", "Badder", "Rosin", "Shatter", "Diamonds"])
  lean, notes = STRAINS[strain]
  low = rng.randrange(70, 88)
  return (
    {
      "name": f"{strain} {form} 1g",
      "category": f"Concentrates > {form}",
      "description": f"One gram of {strain} {form.lower()}, a {lean} extract from fresh-frozen flower with notes "
      f"of {notes}.",
    },
    {"weight_per_item": {"value": 1, "unit": "g"}, "thc_per_item": {"min": low, "max": low + 5, "unit": "%"}},
  )


def tincture(rng: random.Random) -> tuple[dict, dict]:
  purpose = rng.choice(["Daytime", "Nighttime", "Relief", "Balance"])
  flavor = rng.choice(["Peppermint", "Citrus", "Vanilla", "Unflavored"])
  cbd, thc = rng.choice([(300, 300), (600, 300), (1000, 100), (1200, 60)])
  return (
    {
      "name": f"{purpose} {cbd // thc}:1 Tincture {flavor} 30ml",
      "category": "Tinctures > Drops",
      "description": f"A {flavor.lower()} tincture in MCT oil with {cbd}mg CBD and {thc}mg THC per 30 ml bottle, "
      "with a marked dropper.",
    },
    {
      "volume_per_item": {"value": 30, "unit": "ml"},
      "cbd_per_item": {"value": cbd, "unit": "mg"},
      "thc_per_item": {"value": thc, "unit": "mg"},
    },
  )


def topical(rng: random.Random) -> tuple[dict, dict]:
  scent = rng.choice(["Lavender", "Eucalyptus", "Unscented", "Cedar", "Peppermint", "Arnica"])
  form = rng.choice(["Balm", "Lotion", "Salve"])
  cbd = rng.choice([250, 500, 1000])
  return (
    {
      "name": f"{scent} Relief {form} {cbd}mg CBD",
      "category": f"Topicals > {form}s",
      "description": f"A {scent.lower()} {form.lower()} with shea butter and {cbd}mg CBD per jar, for sore muscles "
      "and dry skin. Not psychoactive.",
    },
    {"weight_per_item": {"value": 60, "unit": "g"}, "cbd_per_item": {"value": cbd, "unit": "mg"}},
  )


KINDS = {
  "gummies": gummies, "chocolate": chocolate, "seltzer": seltzer, "cartridge": cartridge, "flower": flower,
  "preroll": preroll, "concentrate": concentrate, "tincture": tincture, "topical": topical,
}

# The products the steps below are about, written out so that the steps do not depend on the seed. They are the
# first of the catalog; the rest is drawn.
HEROES = [
  (
    0,
    {
      "name": "Blueberry Gummies 100mg",
      "category": "Edibles > Gummies",
      "description": "Ten vegan fruit gummies, indica-leaning, with a ripe blueberry flavor. 10mg THC per piece, "
      "100mg per pouch.",
    },
    {"weight_per_item": {"value": 40, "unit": "g"}, "thc_per_item": {"value": 100, "unit": "mg"}},
  ),
  (
    1,
    {
      "name": "Sour Diesel Live Resin Cartridge 1g",
      "category": "Vape Pens > Cartridges",
      "description": "A 1 gram 510-thread cartridge of Sour Diesel live resin, a sativa with notes of citrus and "
      "fuel. No cutting agents.",
    },
    {"weight_per_item": {"value": 1, "unit": "g"}, "thc_per_item": {"min": 84, "max": 88, "unit": "%"}},
  ),
  (
    0,
    {
      "name": "Watermelon Gummies 100mg",
      "category": "Edibles > Gummies",
      "description": "Ten pectin-based fruit gummies, hybrid-leaning, with a ripe watermelon flavor. 10mg THC per "
      "piece, 100mg per pouch.",
    },
    {"weight_per_item": {"value": 40, "unit": "g"}, "thc_per_item": {"value": 100, "unit": "mg"}},
  ),
  (
    2,
    {
      "name": "Blue Dream 3.5g",
      "category": "Flower > Sativa",
      "description": "Indoor-grown Blue Dream, a sativa with dense, frosty buds and notes of blueberry and sweet "
      "herbs. Hand trimmed and slow cured.",
    },
    {"weight_per_item": {"value": 3.5, "unit": "g"}, "thc_per_item": {"min": 22, "max": 25, "unit": "%"}},
  ),
  (
    3,
    {
      "name": "Sea Salt Dark Chocolate Bar 100mg",
      "category": "Edibles > Chocolates",
      "description": "A dark chocolate bar with sea salt, scored into 10 squares of 10mg THC each. 100mg THC per "
      "bar.",
    },
    {"weight_per_item": {"value": 50, "unit": "g"}, "thc_per_item": {"value": 100, "unit": "mg"}},
  ),
]


def demo_id(number: int) -> str:
  return f"demo-{number:04}"


def brand_id_of(brand: int) -> str:
  return f"brand-{brand + 1:02}"


def product(number: int, brand: int, fields: dict, quantities: dict) -> dict:
  name, _ = BRANDS[brand]
  return {
    "id": demo_id(number),
    "brand_id": brand_id_of(brand),
    "fields": {**fields, "brand": name},
    "quantities": quantities,
  }


def catalog(count: int, seed: int) -> list[dict]:
  """`count` products, no two of one brand with the same name. The same seed gives the same catalog."""
  rng = random.Random(seed)
  products = [product(index + 1, brand, fields, quantities) for index, (brand, fields, quantities) in enumerate(HEROES)]
  taken = {(item["brand_id"], item["fields"]["name"]) for item in products}
  attempts = 0
  while len(products) < count and attempts < count * 200:
    attempts += 1
    brand = rng.randrange(len(BRANDS))
    fields, quantities = KINDS[rng.choice(BRANDS[brand][1])](rng)
    item = product(len(products) + 1, brand, fields, quantities)
    if (item["brand_id"], fields["name"]) not in taken:
      taken.add((item["brand_id"], fields["name"]))
      products.append(item)
  return products[:count]


def export_line(item: dict) -> str:
  """One line of the export format `record-similarity import` and `just client import` read."""
  record = {**item["fields"], **item["quantities"]}
  return json.dumps({"id": item["id"], "brand_id": item["brand_id"], "record": record})


# --- talking to the server -----------------------------------------------------------------------------------------


def stubs():
  """The client, generated from the `.proto` the server is built from, into a directory that is gone on exit."""
  out = tempfile.mkdtemp(prefix="record-similarity-demo-")
  status = protoc.main(["protoc", f"-I{PROTO_ROOT}", f"--python_out={out}", f"--grpc_python_out={out}", PROTO_FILE])
  if status != 0:
    raise SystemExit(f"could not generate a client from {PROTO_ROOT / PROTO_FILE}")
  sys.path.insert(0, out)
  return (
    importlib.import_module("recordsimilarity.v1.similarity_pb2"),
    importlib.import_module("recordsimilarity.v1.similarity_pb2_grpc"),
  )


class Paint:
  """Color on a terminal, plain text into a file or a pipe."""

  def __init__(self, enabled: bool) -> None:
    self.enabled = enabled

  def wrap(self, code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

  def bold(self, text: str) -> str:
    return self.wrap("1", text)

  def dim(self, text: str) -> str:
    return self.wrap("2", text)

  def good(self, text: str) -> str:
    return self.wrap("32", text)

  def bad(self, text: str) -> str:
    return self.wrap("31", text)

  def warn(self, text: str) -> str:
    return self.wrap("33", text)


class Demo:
  def __init__(self, pb, stub, paint: Paint, pause: bool, products: list[dict]) -> None:
    self.pb, self.stub, self.paint, self.pause = pb, stub, paint, pause
    self.products = products
    self.names = {item["id"]: f"{item['fields']['name']} ({item['fields']['brand']})" for item in products}
    self.step_number = 0
    self.latencies: list[float] = []

  # -- building requests --

  def quantity(self, value: dict):
    low, high = (value["value"], value["value"]) if "value" in value else (value["min"], value["max"])
    return self.pb.Quantity(min=low, max=high, unit=value["unit"])

  def record(self, item: dict, identified: bool = True):
    quantities = {name: self.quantity(value) for name, value in item["quantities"].items()}
    if identified:
      return self.pb.Record(id=item["id"], brand_id=item["brand_id"], fields=item["fields"], quantities=quantities)
    return self.pb.Record(fields=item["fields"], quantities=quantities)

  # -- printing --

  def step(self, title: str, *lines: str) -> None:
    if self.pause and self.step_number > 0:
      input(self.paint.dim("\n[Enter for the next step] "))
    self.step_number += 1
    print()
    print(self.paint.bold(f"{self.step_number}. {title}"))
    for line in lines:
      print(f"   {line}")
    print()

  def sent(self, label: str, item: dict, was: dict | None = None) -> None:
    """What goes out. With `was`, only what differs from it, so the change is what the audience reads."""
    print(f"   {self.paint.dim('send')}      {label}")
    for name, value in {**item["fields"], **item["quantities"]}.items():
      before = None if was is None else {**was["fields"], **was["quantities"]}.get(name)
      if was is not None and before == value:
        continue
      print(field_line(name, value))

  def took(self, seconds: float) -> str:
    return self.paint.dim(f"{seconds * 1000:.1f} ms")

  def timed(self, call, request):
    started = time.perf_counter()
    response = call(request)
    elapsed = time.perf_counter() - started
    self.latencies.append(elapsed)
    return response, elapsed

  def verdict(self, response, elapsed: float) -> None:
    pb, paint = self.pb, self.paint
    name = pb.Verdict.Name(response.verdict).removeprefix("VERDICT_")
    shown = paint.bad(name) if response.verdict == pb.VERDICT_REJECTED else paint.good(name)
    stored = "stored" if response.stored else "nothing stored"
    print(f"   verdict   {paint.bold(shown)} ({stored}, revision {response.revision})   {self.took(elapsed)}")
    if response.HasField("similarity"):
      print(f"   overall   {response.similarity:.3f} against a threshold of {response.threshold:.3f}")
    if response.HasField("anchor_similarity"):
      print(f"   drift     {response.anchor_similarity:.3f} similarity to the product as first stored")
    for field in response.fields:
      change = pb.FieldChange.Name(field.change).removeprefix("FIELD_CHANGE_").capitalize()
      score = f"{field.similarity:.3f}" if field.HasField("similarity") else "     "
      floor = paint.bad("  below its floor") if field.below_floor else ""
      print(f"   field     {field.field:<16}{score}  {change}{floor}")
    for quantity in response.quantities:
      change = pb.QuantityChange.Name(quantity.change).removeprefix("QUANTITY_CHANGE_").capitalize()
      factor = f"x{quantity.factor:.2f}" if quantity.HasField("factor") else "     "
      units = f" ({quantity.previous_unit} to {quantity.unit})" if quantity.HasField("previous_unit") else ""
      print(f"   quantity  {quantity.field:<16}{factor:<7}{change}{units}")
    for reason in response.reasons:
      print(f"   reason    {paint.bad(reason)}")

  def evaluate(self, item: dict, mode: int):
    response, elapsed = self.timed(self.stub.Evaluate, self.pb.EvaluateRequest(record=self.record(item), mode=mode))
    self.verdict(response, elapsed)
    return response

  # -- the steps --

  def describe(self) -> None:
    self.step("What the server is running", "Describe: the model, the fields it compares, and the limits in force.")
    response, elapsed = self.timed(self.stub.Describe, self.pb.DescribeRequest())
    print(f"   model     {response.model_fingerprint} ({response.dimension} numbers per field)   {self.took(elapsed)}")
    print(f"   accept    an edit that scores {response.threshold:.2f} or more against the stored product")
    if response.HasField("anchor_threshold"):
      print(f"   drift     and {response.anchor_threshold:.2f} or more against the product as first stored")
    for field in response.fields:
      floor = f", rejected below {field.reject_below:.2f}" if field.HasField("reject_below") else ""
      print(f"   field     {field.name:<16}weight {field.weight:g}{floor}")
    print(f"   quantity  {', '.join(quantity.name for quantity in response.quantities)}")

  def forget(self, record_id: str, brand_id: str) -> bool:
    """Delete one product. An earlier run with another seed stored the same id under another brand, and a delete
    has to name the brand a record was stored under, so the other brands are tried when the first is refused."""
    for candidate in [brand_id, *(brand_id_of(brand) for brand in range(len(BRANDS)))]:
      try:
        return self.stub.Delete(self.pb.DeleteRequest(id=record_id, brand_id=candidate)).deleted
      except grpc.RpcError as error:
        if error.code() != grpc.StatusCode.FAILED_PRECONDITION:
          raise
    return False

  def clear(self) -> int:
    """Forget the demo's products, so that a second run starts where the first did. An earlier run may have loaded
    more than this one will, so the ids are walked until a long stretch of them is not stored."""
    removed, missing, number = 0, 0, 0
    while number < len(self.products) or missing < 100:
      number += 1
      brand_id = self.products[number - 1]["brand_id"] if number <= len(self.products) else brand_id_of(0)
      if self.forget(demo_id(number), brand_id):
        removed, missing = removed + 1, 0
      else:
        missing += 1
    return removed

  def load(self, title: str, *lines: str) -> None:
    self.step(title, *lines)
    requests = (self.pb.ImportRequest(record=self.record(item)) for item in self.products)
    outcomes: Counter = Counter()
    failures = []
    started = time.perf_counter()
    for answer in self.stub.Import(requests):
      outcomes[answer.outcome] += 1
      if answer.outcome == self.pb.IMPORT_OUTCOME_FAILED:
        failures.append(f"record {answer.index} ({answer.id}): {answer.message}")
    elapsed = time.perf_counter() - started
    count = len(self.products)
    print(f"   sent      {count} products over one stream in {elapsed:.2f} s ({count / elapsed:.0f} a second)")
    print(f"   created   {outcomes[self.pb.IMPORT_OUTCOME_CREATED]}")
    print(f"   existing  {outcomes[self.pb.IMPORT_OUTCOME_EXISTS]} (left exactly as they were)")
    print(f"   failed    {outcomes[self.pb.IMPORT_OUTCOME_FAILED]}")
    for failure in failures[:5]:
      print(f"             {self.paint.bad(failure)}")

  def ordinary_edit(self) -> None:
    was = self.products[0]
    now = edited(
      was,
      name="Blueberry Gummies 100mg (10-pack)",
      description="Ten vegan fruit gummies with a ripe blueberry flavor, indica-leaning and made with real fruit. "
      "10mg THC per piece, 100mg per pouch.",
    )
    self.step(
      "An ordinary edit",
      f"The brand tidies up {was['id']}: a pack size in the name, a reworded description.",
      "First as a dry run, which judges and stores nothing, then for real.",
    )
    self.sent("the edited product, as a dry run", now, was)
    self.evaluate(now, self.pb.WRITE_MODE_DRY_RUN)
    print()
    self.sent("the same edit, enforced", now, was)
    self.evaluate(now, self.pb.WRITE_MODE_ENFORCE)

  def recycled(self) -> None:
    was = self.products[1]
    now = {
      **was,
      "fields": {
        "name": "Strawberry Lemonade Gummies 200mg",
        "category": "Edibles > Gummies",
        "brand": was["fields"]["brand"],
        "description": "Twenty vegan fruit gummies with a tart strawberry lemonade flavor. 10mg THC per piece, "
        "200mg per pouch.",
      },
      "quantities": {"weight_per_item": {"value": 80, "unit": "g"}, "thc_per_item": {"value": 200, "unit": "mg"}},
    }
    self.step(
      "A recycled product",
      f"{was['id']} is a vape cartridge with a history. The same id now arrives describing another product,",
      "which is how a product's reviews and ranking get carried over to something else.",
    )
    self.sent("a different product under the old id", now, was)
    self.evaluate(now, self.pb.WRITE_MODE_ENFORCE)
    print()
    print("   And the stored product is as it was. The original, sent again as a dry run:")
    print()
    self.evaluate(was, self.pb.WRITE_MODE_DRY_RUN)

  def small_steps(self) -> None:
    was = self.products[3]
    self.step(
      "The same swap, in small steps",
      f"{was['id']} is turned into another strain a little at a time, so that no single edit looks like much.",
      "Every edit is compared with the stored product, and also with the product as it was first stored.",
    )
    for label, now in drifting(was):
      self.sent(label, now, was)
      response = self.evaluate(now, self.pb.WRITE_MODE_ENFORCE)
      print()
      if response.verdict == self.pb.VERDICT_REJECTED:
        break
      was = now

  def quantity_change(self) -> None:
    was = self.products[2]
    now = {**was, "quantities": {**was["quantities"], "thc_per_item": {"value": 1000, "unit": "mg"}}}
    self.step(
      "A number changes and the text does not",
      f"{was['id']} keeps every word and goes from 100 mg of THC to 1000 mg. Quantities are compared as numbers,",
      "beside the text and never inside it, and reported in every answer. No limit on them ships: where a limit",
      "belongs has to be measured on real edits first. This is a dry run.",
    )
    self.sent("the same text with ten times the THC", now, was)
    self.evaluate(now, self.pb.WRITE_MODE_DRY_RUN)

  def wrong_brand(self) -> None:
    was = self.products[4]
    other = next(item["brand_id"] for item in self.products if item["brand_id"] != was["brand_id"])
    now = {**edited(was, name="Sea Salt Dark Chocolate Bar 100mg (new recipe)"), "brand_id": other}
    self.step(
      "An edit from the wrong brand",
      f"{was['id']} belongs to {was['brand_id']}. An edit for it arrives naming {other}.",
    )
    self.sent(f"an edit as {other}", now, was)
    try:
      self.timed(self.stub.Evaluate, self.pb.EvaluateRequest(record=self.record(now), mode=self.pb.WRITE_MODE_ENFORCE))
      print(f"   {self.paint.warn('the server accepted the request')}")
    except grpc.RpcError as error:
      print(f"   refused   {self.paint.bad(error.code().name)}: {error.details()}")

  def match(self) -> None:
    menu_item = {"name": "Blueberry Gummies 10pk 100mg", "brand": "Sunny Orchard", "category": "Edibles"}
    self.step(
      "A retailer's menu item, matched to the catalog",
      "A menu item typed by a store, with no id: which stored product is it? Match reads and never writes.",
      "The server keeps no product text, so the names on the right are this script's own copy.",
    )
    print(f"   {self.paint.dim('send')}      a menu item")
    for name, value in menu_item.items():
      print(field_line(name, value))
    response, elapsed = self.timed(
      self.stub.Match, self.pb.MatchRequest(menu_item=self.pb.Record(fields=menu_item), limit=5)
    )
    print(f"   answered  {len(response.candidates)} candidates, best first   {self.took(elapsed)}")
    for candidate in response.candidates:
      print(f"   {candidate.similarity:.3f}     {candidate.id}  {self.names.get(candidate.id, '')}")
    if response.decision == self.pb.MATCH_DECISION_NOT_JUDGED:
      print("   decision  not judged: no match threshold is configured, and none ships. It has to be measured.")
    else:
      decision = self.pb.MatchDecision.Name(response.decision).removeprefix("MATCH_DECISION_").lower()
      print(f"   decision  {decision.replace('_', ' ')} at a threshold of {response.threshold:.3f}")

  def neighbors(self) -> None:
    item = self.products[2]
    self.step(
      "The nearest products to a stored one",
      f"FindSimilar for {item['id']}, {self.names[item['id']]}: what else in the store looks like it.",
    )
    response, elapsed = self.timed(self.stub.FindSimilar, self.pb.FindSimilarRequest(id=item["id"], limit=6))
    print(f"   answered  {len(response.neighbors)} neighbors   {self.took(elapsed)}")
    for neighbor in response.neighbors:
      print(f"   {neighbor.similarity:.3f}     {neighbor.id}  {self.names.get(neighbor.id, '')}")

  def summary(self) -> None:
    print()
    median = statistics.median(self.latencies) * 1000
    slowest = max(self.latencies) * 1000
    print(
      self.paint.dim(
        f"{len(self.latencies)} single requests from this script: median {median:.1f} ms, slowest {slowest:.1f} ms, "
        "each timed from here and so including the client."
      )
    )


def edited(item: dict, **fields: str) -> dict:
  return {**item, "fields": {**item["fields"], **fields}}


def drifting(item: dict) -> list[tuple[str, dict]]:
  """Edits that take a jar of one strain to a jar of another, each a small change to the one before."""
  first = edited(item, name="Blue Dream 3.5g (Indoor)")
  second = edited(
    first,
    description="Indoor-grown Blue Dream, a sativa with dense, frosty buds and notes of blueberry and sweet herbs. "
    "Now crossed with Gelato.",
  )
  third = edited(second, name="Blue Dream x Gelato 3.5g (Indoor)")
  fourth = edited(
    third,
    description="Indoor-grown Blue Dream crossed with Gelato, a hybrid with dense, frosty buds and notes of sweet "
    "cream and citrus. Hand trimmed and slow cured.",
  )
  fifth = edited(fourth, category="Flower > Hybrid")
  return [
    ("edit 1: a word added to the name", first),
    ("edit 2: a sentence swapped in the description", second),
    ("edit 3: a second strain joins the name", third),
    ("edit 4: the description follows", fourth),
    ("edit 5: one word of the category", fifth),
  ]


def field_line(name: str, value) -> str:
  """One field of a request, wrapped so that a description stays on the slide."""
  text = value if isinstance(value, str) else json.dumps(value)
  indent = " " * 13
  return textwrap.fill(f"{name}: {text}", width=110, initial_indent=indent, subsequent_indent=indent + "  ")


def channel(server: str, allow_remote: bool):
  url = urlparse(server if "://" in server else f"http://{server}")
  host, port = url.hostname or "127.0.0.1", url.port or 50051
  if host not in ("127.0.0.1", "localhost", "::1") and not allow_remote:
    raise SystemExit(
      f"{host} is not this machine. The demo creates and deletes `demo-NNNN` products on the server it is given: "
      "pass --allow-remote if that server is one made for it."
    )
  target = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
  if url.scheme == "https":
    return grpc.secure_channel(target, grpc.ssl_channel_credentials()), target
  return grpc.insecure_channel(target), target


def main() -> int:
  parser = argparse.ArgumentParser(description="A walk through record-similarity against a running server.")
  parser.add_argument("--server", default="http://127.0.0.1:50051", help="the server's address")
  parser.add_argument("--products", type=int, default=300, help="how many synthetic products to load")
  parser.add_argument("--seed", type=int, default=7, help="the same seed makes the same catalog")
  parser.add_argument("--pause", action="store_true", help="wait for Enter before each step")
  parser.add_argument("--export", type=Path, help="also write the catalog as an export file for `import`")
  parser.add_argument("--cleanup", action="store_true", help="delete the demo's products from the server and stop")
  parser.add_argument("--allow-remote", action="store_true", help="let --server be another machine")
  parser.add_argument("--no-color", action="store_true", help="plain text even on a terminal")
  args = parser.parse_args()
  if args.products < len(HEROES):
    parser.error(f"--products must be at least {len(HEROES)}: the steps are about the first {len(HEROES)}")

  products = catalog(args.products, args.seed)
  if args.export:
    args.export.write_text("".join(export_line(item) + "\n" for item in products))

  link, target = channel(args.server, args.allow_remote)
  pb, rpc = stubs()
  paint = Paint(sys.stdout.isatty() and not args.no_color)
  demo = Demo(pb, rpc.RecordSimilarityStub(link), paint, args.pause, products)
  try:
    grpc.channel_ready_future(link).result(timeout=5)
  except grpc.FutureTimeoutError:
    print(f"no server answered at {target}. Start one with `just serve-in-memory` and run this again.", file=sys.stderr)
    return 1

  try:
    removed = demo.clear()
    if args.cleanup:
      print(f"deleted {removed} demo products from {target}")
      return 0
    print(paint.bold(f"record-similarity, at {target}"))
    if removed:
      print(paint.dim(f"({removed} products of an earlier run of this demo were deleted first)"))
    demo.describe()
    demo.load(
      "Loading a catalog",
      f"{len(products)} synthetic products from {len({item['brand_id'] for item in products})} brands, sent over one "
      "Import stream.",
      "The server turns each product's fields into vectors and stores those: it keeps no product text.",
    )
    demo.load(
      "Loading it again",
      "An import only creates. A product that is stored is left alone, so a load can be repeated or resumed.",
    )
    demo.ordinary_edit()
    demo.recycled()
    demo.small_steps()
    demo.quantity_change()
    demo.wrong_brand()
    demo.match()
    demo.neighbors()
    demo.summary()
  except grpc.RpcError as error:
    print(f"the server refused a request: {error.code().name}: {error.details()}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
