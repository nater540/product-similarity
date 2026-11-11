# Hand-off prompt: calibration data for record-similarity

Paste everything below the line into a Claude session that has read access to the product data.

---

I need your help assembling a labeled dataset from our product listing data. You will be working inside our
environment with read access to it; I cannot move this data elsewhere, so everything you produce stays here.

## Why

We have a service, `record-similarity`, that decides whether an edit to a product listing is still the same product.
The problem it targets is "recycling": a merchant takes an existing listing (say a blueberry gummy) and edits it
into an entirely different product, keeping the listing's id, reviews, and history.

The service embeds each text field of a listing with a small sentence-embedding model, compares the new version
with the stored one field by field (cosine similarity), and rejects the edit if the weighted similarity falls below
a threshold. Today those thresholds are calibrated on 46 hand-written, synthetic examples. I need real before/after
pairs, with trustworthy labels, to replace them. The dataset will be used to choose thresholds and field weights,
and possibly later to fine-tune the model.

## Start by finding out what exists

Before pulling anything at scale, find out and tell me:

- Whether there is a version history or audit log of product listing text (what a listing said before and after
  each edit, with timestamps). If there is not, stop and report that: nothing below is possible without it.
- Which fields a listing has, which are free text and which are ids or enums, and how often each is filled in.
- What independent evidence of recycling exists (see "Labels").

If you cannot find where something lives, ask me rather than guessing. If reaching a source needs access you do not
have, stop and tell me; do not work around it.

## What I need, in priority order

Write everything into one new directory outside any git repository.

### 1. `pairs.full.jsonl`: labeled before/after pairs (the source of truth)

One JSON object per line, one line per edit:

```json
{"pair_id": "p000123", "product": "<pseudonym>", "merchant": "<pseudonym>", "edited_at": "2026-03-14T17:02:11Z",
 "label": "recycle", "label_source": "moderator_revert", "label_confidence": "high", "variant_change": null,
 "split": "calibrate", "note": "gummies listing became a vape cartridge",
 "before": {"name": "...", "category": "...", "brand": "...", "description": "...", "<other field>": "..."},
 "after":  {"name": "...", "category": "...", "brand": "...", "description": "...", "<other field>": "..."}}
```

- `before` is the version immediately preceding the edit, `after` the version it produced.
- Include **every** field that describes what the product is, as text: name, category (the human-readable path such
  as `Edibles > Gummies`, not the id), brand name, description, and anything else we have (strain, genetics, size or
  weight, potency, flavor, tags). Where a field is an id or enum, give its display text and add the raw id under a
  second key with an `_id` suffix.
- Omit a field that has no value. Do not write `null` or an empty string.
- **Do not clean the text.** No trimming, lowercasing, HTML stripping, translation, de-duplication of whitespace, or
  encoding repair. The service must be calibrated on exactly what it will receive.

### 2. `pairs.jsonl`: the same pairs, in the strict format the calibration tool reads

Derived from file 1 by projection. The tool rejects any line that deviates from this:

```json
{"label": "edit", "note": "p000123 potency added to the name", "before": {"name": "..."}, "after": {"name": "..."}}
```

- Exactly these four keys. `label` is `"edit"` or `"recycle"`, lowercase.
- `before` and `after` may contain only the keys `name`, `category`, `brand`, `description`, with string values.
- `name` must be present and non-blank on both sides; leave out any pair where it is not, and count how many.
- Start `note` with the `pair_id` so a line can be traced back to file 1.
- Include only pairs with `label_confidence` of `high` or `medium`, and only `split: "calibrate"`.
- Blank lines and lines starting with `#` are allowed.

### 3. `sequences.jsonl`: whole edit histories

For about 200 listings that have had at least five text edits, every version in order. Include listings known to
have been recycled if any exist. This is used to measure how far a legitimate listing drifts over its lifetime.

```json
{"product": "<pseudonym>", "label": "edit", "versions": [{"at": "2025-11-02T09:00:00Z", "fields": {"name": "..."}}]}
```

### 4. `catalog-groups.jsonl` (only if this link exists)

If merchant listings are linked to a shared catalog or verified brand product, export about 2,000 groups: the
catalog product's fields plus the fields of up to ten merchant listings linked to it. Different merchants describing
the same product are naturally occurring paraphrases, which is training data for the model. If no such link exists,
say so and skip this.

```json
{"group": "<pseudonym>", "catalog": {"name": "..."}, "listings": [{"name": "..."}, {"name": "..."}]}
```

### 5. `REPORT.md`

- What you found in "Start by finding out what exists", including the queries you ran.
- Counts by label, label source, confidence, top-level category, and split.
- An estimate of the base rate: of a uniform random sample of text edits, what fraction look like recycling.
- Per field: fill rate, typical and maximum length, share of non-English text.
- Counts of what you excluded and why.
- Known biases in the sample, and anything you could not do.
- Twenty borderline examples (see "Variant changes") for a human to rule on.

## Labels

- `edit`: after the change the listing still describes the same physical product (reworded, corrected,
  reformatted, extended, potency or batch updated).
- `recycle`: after the change the listing describes a different product.

**The label must come from evidence independent of how different the text looks.** If you label by reading the two
versions and judging how much changed, the calibration becomes circular and worthless. Acceptable evidence, roughly
from strongest to weakest:

1. A moderator or support action: the edit was reverted or the listing was flagged or removed for this reason.
2. A change of identity in structured data at the same time: barcode, SKU, lab batch or product registration id,
   linked catalog product, product type.
3. Customer evidence: reviews or tickets after the edit saying the listing is not the product it used to be.
4. A discontinuity around the edit in orders, price, or product photos.

For `edit` labels use positive evidence too where you can (identifiers unchanged, no complaints, no moderation
action), not merely the absence of a recycle signal.

Record which evidence you used in `label_source` and how strong it is in `label_confidence` (`high`, `medium`,
`low`). If the evidence is absent or conflicting, set `label` to `"unsure"` in file 1 and leave the pair out of
file 2. Never guess, and never invent or synthesize pairs to fill a quota.

### Variant changes

Some edits are a matter of policy rather than fact: the same brand and category, but a different flavor, strain,
size, pack count, or potency. Set `variant_change` to one of `"flavor"`, `"strain"`, `"size"`, `"count"`,
`"potency"` for these (otherwise `null`). Label them only if a written policy or a moderator decision settles the
question; otherwise mark them `"unsure"`. I want as many of these as you can find, because they are where the
threshold will actually sit.

## Sampling

- Edits from the last 12 months. Only edits that changed the text of at least one descriptive field; skip edits to
  price, stock, or photos alone, and edits whose text is identical apart from whitespace. Count what you skip.
- Aim for at least 300 `edit` and 300 `recycle` pairs at `high` or `medium` confidence. Recycles are rare: take
  every well-evidenced one you can find, up to about 1,000, and tell me if you fall short.
- Spread across every top-level category roughly in proportion to edit volume, with a minimum of 20 pairs for any
  category that has them.
- At most five pairs per merchant and one pair per listing in files 1 and 2, so that no single merchant's writing
  style dominates.
- In addition, a uniform random sample of 500 text edits, labeled as well as the evidence allows, tagged
  `"label_source": "random_sample"` plus the evidence used. This is what the base-rate estimate comes from.
- Assign `split`: 80% `"calibrate"`, 20% `"holdout"`, decided by a hash of the merchant so that no merchant appears
  in both.

## Handling

- Read-only. Do not write to any production system.
- Keep the data inside this environment. Do not send listing text to any external service, API, or tool.
- Replace product and merchant ids with a salted hash, stable across all files, and do not write the salt into any
  output. No customer names, emails, or other personal data in any file; use review and ticket text as evidence for
  labels, but do not export it.
- If anything here conflicts with our data-handling policy, follow the policy and tell me what you changed.

## Before you finish

- Verify that every line of every `.jsonl` file parses and has exactly the keys specified, and that every line of
  `pairs.jsonl` meets the strict rules above.
- If the `record-similarity` binary and its model are available here, run
  `record-similarity calibrate --config config.example.toml pairs.jsonl` and paste its output into the report.
- Finish with a short summary: what you produced, the counts, what fell short of the targets, and the decisions you
  need from me.
