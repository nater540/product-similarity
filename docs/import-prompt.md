# Hand-off prompt: loading the real catalog into record-similarity

Paste everything below the line into a Claude session that has read access to the product data and can reach the
`record-similarity` deployment. It is a second brief, separate from `docs/data-request-prompt.md`: that one
collects labeled pairs for choosing thresholds, and its files are pseudonymized. This one loads the catalog into
the running service, and uses real ids, because the service has to recognize them later.

---

I need your help loading our brand products into `record-similarity`, so that every product has a baseline there
before the service starts judging edits. You will be working inside our environment; the product data and the
service both stay here.

## Scope

**Brand products**: the catalog entries a brand owns and maintains. Not retailers' menu items. Wherever this says
"product" it means a brand product, and "brand" means the brand that owns it.

## What the service is, and what loading it means

`record-similarity` decides whether an edit to a product is still the same product. It keeps, for every product,
a *baseline* (vectors of the product's text fields, a fingerprint of each field's text, and its quantities) and an
*anchor* (the first value each field ever had, for measuring how far a product has drifted). It never stores the
text itself. When an edit arrives it is compared with the baseline.

A product the service has never seen has no baseline, so its first edit is simply accepted and becomes one.
Loading the catalog first is what makes the first real edit of every product a judged one.

Loading is done with the service's **import**, which has one rule above all others: **it only creates.** A
product id that is already stored is left exactly as it is. An import cannot replace a baseline, cannot move an
anchor, and can be run again at any time: the second run creates what is missing and touches nothing else.

## Start by finding out, and tell me before you load anything

- **Which deployment.** The server's address, and whether there is a staging one. Ask me if you are not sure
  which is which. Then ask the server what it is running with:

  ```sh
  just client --server http://HOST:PORT describe
  ```

  (From a checkout of the `record-similarity` repository. `just client ...` is
  `cargo run --release -q -p similarity-server --example client -- ...`.) It prints the model fingerprint, the
  text fields with their weights, and the quantities. **The field and quantity names it prints are the only keys
  a record may have.** Send me that output.
- **How the application that will call the service builds a product's record**, if that integration exists
  already: which column becomes which field, how the category text is put together, what it sends as the
  product's id and as its brand. **The import has to build every record exactly the same way.** If the import
  writes `Edibles > Gummies` and the application later sends `Gummies, Edibles`, every product looks edited the
  first time it is saved. If the integration does not exist yet, tell me, and use the mapping below; whoever
  builds it must then follow the same mapping.
- **How many products there are**, how many brands, and whether inactive, archived, or deleted products should be
  loaded. My default: every product that can still be edited. Ask if that is not a clear line in the data.
- **Whether the product as it was first created can be recovered** (a version history or audit log). If it can,
  each record can carry it as its `anchor` and keep its real drift history. If it cannot, say so: every product's
  history then starts on the day of the import.
- **Whether anything is stored already.** After a first load into an empty collection I expect every record to
  be created. If the service says many already exist, stop and tell me before going on.

If reaching something needs access you do not have, stop and tell me. Do not work around it.

## The export file

One JSON object per line, one line per product, in a new directory outside any git repository:

```json
{"id": "8841207", "brand_id": "5512", "record": {"name": "Blueberry Gummies 100mg", "category": "Edibles > Gummies", "brand": "Sunny Orchard", "description": "Ten vegan gummies...", "weight_per_item": {"value": 40, "unit": "g"}, "thc_per_item": {"min": 100, "max": 100, "unit": "mg"}}, "anchor": {"name": "Blueberry Gummies", "category": "Edibles > Gummies", "brand": "Sunny Orchard"}}
```

Exactly these four keys; any other top-level key is refused, so that a misspelled one is noticed.

- **`id` is the product's real id, as a string, exactly as the application will send it** when it asks about an
  edit. Not a pseudonym, not a hash. If the application will send `"8841207"`, the import sends `"8841207"`.
- **`brand_id` is the key of the brand that owns the product, as a string.** Once a product is stored under a
  brand, every later request about it must name the same brand, or it is refused. So this too has to be
  exactly what the application will send. If the application will send no brand, leave the key out for every
  product. Ask me which it is; do not guess, because it cannot be changed by importing again.
- **`record` is the product as it is now.** `anchor`, optional, is the product as it was first created.
  Both use only the field and quantity names `describe` printed. With the shipped configuration those are:
  - text, as strings: `name`, `category`, `brand`, `description`;
  - quantities, as objects: `weight_per_item`, `volume_per_item`, `thc_per_item`, `cbd_per_item`.
- **`name` is required** and must not be blank. A product without one cannot be loaded; count them.
- **`category` is the display path, root first**: the root category's name, then ` > `, then the sub category's
  name if there is one (`Edibles > Gummies`). Names, never ids. If a product has several category rows at one
  level, join them in a fixed order (alphabetical by name), the same order every time.
- **`brand` is the brand's name**, looked up from the product's brand key.
- **A quantity is an object, never text.** A single value is `{"value": 3.5, "unit": "g"}`; a range is
  `{"min": 18, "max": 22, "unit": "%"}`. The numbers are copied from the columns without rounding or conversion,
  and `unit` is the unit column's value exactly as stored. Do not convert units and do not correct one you think
  is wrong: the service converts the units it knows and keeps any other as it is.
  - If only one of `min` and `max` is set, write it as both.
  - Leave a quantity out when its value is null. A stored zero is a value: keep it.
  - A value with no unit cannot be used: leave that quantity out, and count how many there were.
  - Numbers must be finite and not negative, and `min` not above `max`. A product whose quantity breaks that is
    refused whole; count those and show me a few.
- **Leave out a field that has no value.** No `null`, no empty string.
- **Do not clean the text.** No trimming, lowercasing, HTML stripping, or encoding repair. The service must hold
  exactly what it will later be sent.
- **No other keys inside `record` or `anchor`.** A field the service's schema does not define makes that product
  fail. If we have a field that matters and the schema lacks it (strain, flavor), tell me; do not fold it into
  another field.

This file holds real ids and real product text. It stays in this environment, outside any repository, and is
deleted when the load is verified unless our data-handling policy says otherwise.

## Check the file before anything is loaded

```sh
record-similarity import --config CONFIG.toml export.jsonl --check
```

That reads every line against the schema and loads nothing: no model, no connection, a second or two. It needs
the same configuration file the server runs with; ask me for it. Each product it would refuse is logged with its
line, its id, and the reason, and the command ends with a non-zero status if there was any:

```json
{"level":"WARN","message":"record not imported","line":3,"id":"8841209","reason":"invalid record: unknown field `strain`; the schema has: brand, category, description, name"}
{"level":"INFO","message":"finished","checked_only":true,"records":6,"imported":0,"skipped":0,"failed":4,"elapsed_s":0.0}
```

Fix what is a mistake of the export, count what is a fact about the data, and run it again until what is left is
only the second kind. Tell me the counts by reason before going on.

## Load a little first

Take about a hundred products from several brands and load only those (below). Then, for five of them, ask the
service about the product as it is now, without storing anything:

```sh
just client --server http://HOST:PORT evaluate 8841207 product.json --brand-id 5512 --dry-run
```

where `product.json` is that product's `record` object, built the way the application builds it (through the
application's own code, if the integration exists). The answer has to be:

```text
verdict   Accepted (stored: false, revision 1)
overall   1.000 against a threshold of 0.900
drift     0.969 similarity to the original record
field     brand           1.000  Unchanged
field     category        1.000  Unchanged
field     description     1.000  Unchanged
field     name            1.000  Unchanged
quantity  thc_per_item        x1.00  Unchanged (mg to mg)
quantity  weight_per_item     x1.00  Unchanged (g to g)
```

Every field and every quantity `Unchanged`, overall 1.000, revision 1. **Anything else means the import and the
application build the record differently**, and has to be understood before the rest is loaded: a field that
shows `Modified`, a quantity that shows `Added`, `Removed`, or `Incomparable`, or an error saying the record
`belongs to brand ...`. The `drift` line is the one that may be below 1.000, for a product loaded with an
`anchor`: it is how far the product has already moved from its first version.

## Load the catalog

There are two ways. Tell me which you can use; if both, prefer the first when the catalog is large.

**The command, beside the store.** It needs the model directory and a connection to Qdrant (the API key in
`QDRANT_API_KEY`), and not the server. It has the model to itself, so it does not slow anyone down.

```sh
record-similarity import --config CONFIG.toml export.jsonl
```

It can run while the application is sending edits: the command and the server both store a record only if
nobody has stored it since they looked, so an id both see at the same moment is created once and the import
leaves it alone.

**The stream, through the server.** For when you can reach the server and not the store.

```sh
just client --server http://HOST:PORT import export.jsonl
```

It sends every line over one `Import` call and prints how many were created, how many already existed, and each
one that failed with its reason:

```text
failed    record 2 (8841209): invalid record: unknown field `strain`; the schema has: brand, category, description, name
records   5 sent in 0.31 s
created   2
existing  0 (left as they were)
failed    3
```

`record 2` is the third product sent: the stream counts from 0, where `--check` counts lines from 1. A line that
is not an export line at all (a misspelled top-level key) stops the client before it sends anything. Any gRPC
client can make the same call (`Import(stream ImportRequest) returns (stream ImportResponse)`, in
`similarity.proto`).

- **A stream shares the server's model with everyone else.** While it runs, a save by anyone else waits behind
  it (with one model session, which is the default). On a 12-core development machine, on an earlier and slower
  build, a stream created about 400 products a second, and an ordinary edit took a median of 18 ms beside it where
  it took 2 ms alone. Those are that machine's numbers; time the first thousand
  here and tell me what you see. If the service is already taking edits, load outside editing hours, or ask me
  about lowering `server.import_batch_records`.
- A product that fails does not stop the stream. One that is already stored is left alone and counted.
- If the stream is cut off, send the whole file again. What is stored costs a read and nothing more.

Either way: **do not use `Evaluate` to load products, and never send `WRITE_MODE_FORCE` or `Delete`.** If a
stored product turns out to be wrong, tell me. Do not delete it and load it again: that is the one thing here
that changes a product's history, and it is my decision.

## Verify

- The counts add up: created plus already stored plus failed is the number of lines, and failed is what
  `--check` predicted.
- Load the same file once more. Everything is now "already stored", nothing is created, and it takes seconds.
- Audit the store against the file. This needs the configuration file and a route to Qdrant, like
  `record-similarity import`; if you loaded over the server's stream and have neither, say so and skip it.

  ```sh
  record-similarity audit --config CONFIG.toml export.jsonl --exported-at SECONDS
  ```

  `SECONDS` is when you took the export, as `date +%s` prints it. The audit reads the file and the collection,
  embeds nothing, and writes nothing. Right after a load it should say that every product agrees:
  `0  differences`. A product that was edited through the service since you took the export is listed and not
  counted. Anything else it lists (in the export and not stored, stored and not in the export, stored under
  another brand, text or quantities that differ, stored under another layout, not readable) send to me with
  the whole report. Do not repair any of it: no `Delete`, no `WRITE_MODE_FORCE`.
- Repeat the dry-run check on twenty products you did not check before, from as many brands, including some with
  an `anchor`, some with every quantity, and some with none.
- `describe` still prints the fingerprint it printed at the start.

## What I want back

- The `describe` output, and which deployment it was.
- The mapping you used, column by column, and whether it came from the application's code or from this brief.
- Counts: products in the catalog, lines in the export, created, already stored, failed by reason, left out
  before the export and why.
- How many products had an `anchor`, and why the rest did not.
- For each quantity: how often it was present, how often it was left out for having no unit, and every distinct
  unit string that occurs with a count. A unit the service does not convert is still stored, and is then
  comparable only with itself.
- How long the load took, how it was run, and what an edit's latency was meanwhile if anyone was editing.
- The result of every dry-run check, and anything that was not `Unchanged`.
- The audit's report, whole, if you could run it.
- What you could not do, and the decisions you need from me.

## Handling

- Read-only on the product data. The only thing you write to is the `record-similarity` deployment I named, and
  only through its import.
- Keep product data inside this environment. `record-similarity` runs here and is ours; do not send product text
  to any other service, API, or tool.
- No customer names, emails, or other personal data in the export. A product's record is its catalog fields.
- Loading baselines does not make the service reject anything. Whether an edit is refused is up to how the
  application calls it, and it stays in `DRY_RUN` until the thresholds have been measured on real labeled pairs.
- If anything here conflicts with our data-handling policy, follow the policy and tell me what you changed.
