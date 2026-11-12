# Hand-off prompt: calibration data for record-similarity

Paste everything below the line into a Claude session that has read access to the product data.

---

I need your help assembling a labeled dataset from our brand product data. You will be working inside our
environment with read access to it; I cannot move this data elsewhere, so everything you produce stays here.

## Scope

This is about **brand products**: the catalog entries a brand owns and maintains. It is not about retailers' menu
items. Menu items appear in exactly two places below, both optional: as evidence for a label, and as examples
linked to a brand product (section 4). Wherever this says "product" it means a brand product, and wherever it says
"brand" it means the brand that owns one.

## Why

We have a service, `record-similarity`, that decides whether an edit to a brand product is still the same product.
The problem it targets is "recycling": a brand takes an existing product (say a blueberry gummy) and edits it into
an entirely different product, keeping the product's id, its reviews, its history, and whatever is linked to it.

The service embeds each text field of a product with a small sentence-embedding model, compares the new version
with the stored one field by field (cosine similarity), and rejects the edit if the weighted similarity falls below
a threshold. Today those thresholds are calibrated on 46 hand-written, synthetic examples. I need real before/after
pairs, with trustworthy labels, to replace them. The dataset will be used to choose thresholds and field weights,
and to fine-tune the model.

The service also compares a product's **quantities** (its weight, volume, THC, and CBD) as numbers, after
converting units: it reports how many times larger or smaller each became, and whether its unit can still be
compared with the old one. It can reject on those too, but ships with no limits, because nobody has measured where
a limit belongs. This dataset is what will be measured. So the quantities matter as much as the text here, and
they must arrive as numbers with their units, never rendered into text.

One thing follows from the scope and shapes everything below: **every recycle happens inside one brand.** The
brand does not change, and often the category does not either. The hard cases are a brand turning one of its own
products into another of its own products, and those are the cases I most need.

## Start by finding out what exists

Before pulling anything at scale, find out and tell me:

- Whether there is a version history or audit log of brand product text (what a product said before and after
  each edit, with timestamps). If there is not, stop and report that: nothing below is possible without it.
- Which fields a brand product has, which are free text and which are ids or enums, and how often each is filled
  in. I expect these twelve columns among them; tell me which exist, under what names, and what else is there:

  ```text
  name                        description
  weight_per_item_value       weight_per_item_unit
  volume_per_item_value       volume_per_item_unit
  thc_per_item_min_value      thc_per_item_max_value      thc_per_item_unit
  cbd_per_item_min_value      cbd_per_item_max_value      cbd_per_item_unit
  ```

- For each of the four unit columns, **every distinct value that occurs, exactly as stored, with a count** (`g`,
  `G`, `grams`, `mg`, `oz`, `%`, `mg/g`, an empty string, and so on). The service converts the units it knows and
  treats any other as comparable only with itself, so I need the real vocabulary to know what it must convert.
- For the quantity columns: how often each is null, how often it is zero (and whether zero means "none" or "not
  measured", if you can tell), how often only one of a `min` and `max` pair is set, and whether THC or CBD is ever
  recorded per package rather than per item. Say whether there is a pack-size or item-count column as well.
- How a product's brand and its categories are stored. I expect the brand to be a foreign key and the categories
  to come through a join table (a root category and possibly a sub category); tell me if it is otherwise, and
  whether a product can have more than one category row at a level.
- How many brands and how many brand products there are, and how edits are distributed across brands (a few
  brands making most of the edits changes how the sampling below should be capped).
- What independent evidence of recycling exists (see "Labels").
- Whether retailers' menu items are linked to brand products, and how (see section 4).

If you cannot find where something lives, ask me rather than guessing. If reaching a source needs access you do not
have, stop and tell me; do not work around it.

## What I need, in priority order

Write everything into one new directory outside any git repository.

### 1. `pairs.full.jsonl`: labeled before/after pairs (the source of truth)

One JSON object per line, one line per edit to a brand product:

```json
{"pair_id": "p000123", "product": "<pseudonym>", "owner": "<pseudonym>", "edited_at": "2026-03-14T17:02:11Z",
 "label": "recycle", "label_source": "moderator_revert", "label_confidence": "high", "variant_change": null,
 "split": "calibrate", "note": "gummies product became a vape cartridge",
 "before": {"name": "...", "category": "Edibles > Gummies", "brand": "...", "description": "...",
            "weight_per_item": {"value": 40, "unit": "g"},
            "thc_per_item": {"min": 100, "max": 100, "unit": "mg"}, "<other field>": "..."},
 "after":  {"name": "...", "category": "Vape Pens > Cartridges", "brand": "...", "description": "...",
            "weight_per_item": {"value": 1, "unit": "g"},
            "thc_per_item": {"min": 85, "max": 90, "unit": "%"}, "<other field>": "..."}}
```

- `before` is the version immediately preceding the edit, `after` the version it produced.
- `owner` is the brand that owns the product, as a pseudonym (see "Handling"). It is a separate key from the
  `brand` field inside `before` and `after`, which is the brand's name as text.
- Include **every** field that describes what the product is, as text: name, category, brand name, description,
  and anything else we have (strain, genetics, flavor, tags). Where a field is an id or enum, give its display
  text and add the raw id under a second key with an `_id` suffix.
- **`category` is the display path, built root first**: the root category's name, then ` > `, then the sub
  category's name if there is one (`Edibles > Gummies`). Names, never ids. If a product has several category rows
  at one level, join them in a fixed order (alphabetical by name) and tell me how common that is. The order has to
  be stable, or a product whose rows merely came back in another order looks edited.
- **`brand` is the brand's name**, looked up from the product's brand key, as text.
- **The four quantities are objects, not text**, under exactly these keys: `weight_per_item`, `volume_per_item`,
  `thc_per_item`, `cbd_per_item`.
  - A single value is `{"value": 3.5, "unit": "g"}`. A range is `{"min": 18, "max": 22, "unit": "%"}`.
  - `value`, `min`, and `max` are JSON numbers, copied from the column without rounding or conversion. `unit` is
    the unit column's value exactly as stored. Do not convert units, and do not "fix" a unit you think is wrong.
  - Never render a quantity into text (`"3.5 g"`), and never fold it into `name` or `description`. Whatever the
    name already says about size or potency stays in the name as it is.
  - If only one of `min` and `max` is set, write it as both, and add the key `"<quantity>_one_sided": true` beside
    it so I can count them.
  - Leave a quantity out entirely when its value column is null. A stored zero is a value: keep it.
  - A value with no unit cannot be used. Leave it out and count how many there were, per quantity.
- **Include the brand name even though it will almost always be the same on both sides.** The service keeps it as
  a field. If you find edits where a product's brand did change, do not drop them: count them and tell me, because
  that should be rare and I want to know what it means when it happens.
- Omit a field that has no value. Do not write `null` or an empty string.
- **Do not clean the text.** No trimming, lowercasing, HTML stripping, translation, de-duplication of whitespace, or
  encoding repair. The service must be calibrated on exactly what it will receive.

### 2. `pairs.jsonl`: the same pairs, in the strict format the calibration tool reads

Derived from file 1 by projection. The tool rejects any line that deviates from this:

```json
{"label": "edit", "note": "p000123 potency added to the name", "before": {"name": "..."}, "after": {"name": "..."}}
```

- Exactly these four keys. `label` is `"edit"` or `"recycle"`, lowercase.
- `before` and `after` may contain only these keys:
  - `name`, `category`, `brand`, `description`, with string values;
  - `weight_per_item`, `volume_per_item`, `thc_per_item`, `cbd_per_item`, with object values, each either
    `{"value": <number>, "unit": "<text>"}` or `{"min": <number>, "max": <number>, "unit": "<text>"}` and nothing
    else. Numbers must be finite and not negative, `min` not above `max`, and `unit` not blank. Drop the
    `_one_sided` marker here; the tool refuses keys it does not know.
- `name` must be present and non-blank on both sides; leave out any pair where it is not, and count how many.
- Start `note` with the `pair_id` so a line can be traced back to file 1.
- Include only pairs with `label_confidence` of `high` or `medium`, and only `split: "calibrate"`.
- Blank lines and lines starting with `#` are allowed.

### 3. `sequences.jsonl`: whole edit histories

For about 200 brand products that have had at least five text edits, every version in order, spread across as
many brands as you can. Include products known to have been recycled if any exist. This is used to measure how far
a legitimate product drifts over its lifetime.

```json
{"product": "<pseudonym>", "owner": "<pseudonym>", "label": "edit",
 "versions": [{"at": "2025-11-02T09:00:00Z", "fields": {"name": "..."}}]}
```

- `label` is `edit` for a product that stayed the same product throughout, `recycle` for one that became another,
  and `unsure` otherwise. `label_source` and `label_confidence` mean what they mean in `pairs.full.jsonl`.
- For a `recycle`, add `"recycled_at": N` where you can tell: the index in `versions` (from 0) of the first
  version that is no longer the original product. The versions before it are then a legitimate history.
- If `versions[0]` is not the product as it was first created (older versions were not kept, or the history was
  cut for length), add `"original": {...}` with the fields it was created with if those are known, and
  `"truncated": true` if they are not.
- `fields` uses the same keys as a record in `pairs.full.jsonl`, quantities included.
- No other top-level keys: an unknown key is refused, so that a misspelled one is noticed.

### 4. `catalog-groups.jsonl` (only if menu items are linked to brand products)

If retailers' menu items are linked to the brand product they are a listing of, export about 2,000 groups: one
brand product's fields, plus the fields of up to ten menu items linked to it, preferably from ten different
retailers. Different retailers describing the same product are naturally occurring paraphrases of it, which is
training data for the model. If no such link exists, say so and skip this.

```json
{"group": "<pseudonym>", "owner": "<pseudonym>", "split": "calibrate", "catalog": {"name": "..."},
 "listings": [{"name": "..."}, {"name": "..."}]}
```

- `catalog` is the brand product and `listings` are the menu items. Keep those two key names; the training tool
  reads them.
- Use the same fields and the same no-cleaning rule as file 1, for both the product and the menu items.
- Take the product's text as it was when the menu items were linked to it, if that can be determined. A product
  that was recycled after its menu items were linked would otherwise be paired with listings of what it used to be.
  If it cannot be determined, leave out products with a `recycle` label in file 1 and say that you did.
- Only use links you have reason to trust (made by the retailer or verified, not guessed by a matching job), and
  tell me which kind they are.
- Spread the groups across brands: no brand should supply more than about 5% of them. Include several products
  from the same brand and category where you can (three flavors of one gummy line, say). Telling those apart is
  the skill the model most needs.
- Give every group a `split`, assigned by the same brand hash as file 1 (see "Sampling"), and **include** the
  groups of the `holdout` brands, marked `holdout`. The training tool leaves those out; they are what the tool
  that measures matching (`record-similarity calibrate-match --split holdout`) is run on. `group` must be unique
  in the file: it is the product's id in that measurement.
- Map a menu item's columns onto the same field names as the product's, and tell me how: which menu-item column
  became `name`, `category`, `brand`, and `description`, and whether a menu item's quantities are per item or per
  package. Leave out a field that has no counterpart; do not fill it from the product.
- Tell me, from a uniform sample of menu items, what fraction are linked to no brand product, and whether that
  is because no such product exists or because nobody linked it. The groups contain only menu items that have a
  product, so they cannot say how often a menu item should match nothing.
- Tell me whether a different size or pack of one product (a 1 g and a 3.5 g jar, a single and a five-pack) is
  a different brand product in your catalog. That decides whether a quantity should ever stop a match.
- Optional: the brand product catalog itself, in the format of the import export (`{"id", "tenant", "record"}`,
  pseudonymized the same way), so that listings can be scored against a catalog of the real size.

### 5. `REPORT.md`

- What you found in "Start by finding out what exists", including the queries you ran.
- Counts by label, label source, confidence, top-level category, and split.
- How the pairs are distributed across brands: how many brands contribute, and the largest share any one has.
- An estimate of the base rate: of a uniform random sample of text edits to brand products, what fraction look
  like recycling.
- Per field: fill rate, typical and maximum length, share of non-English text.
- Per quantity: fill rate, the table of distinct unit strings with counts, the share of zeros, the share written
  from one end only, and the smallest and largest value per unit (outliers such as a weight of 35000 in `g` are
  worth knowing about before a limit is chosen).
- Among `edit` pairs and among `recycle` pairs separately: how often each quantity's unit changed, and how often a
  quantity was present on one side only.
- Counts of what you excluded and why.
- Known biases in the sample, and anything you could not do.
- Twenty borderline examples (see "Variant changes") for a human to rule on.

## Labels

- `edit`: after the change the product is still the same physical product (reworded, corrected, reformatted,
  extended, potency or batch updated).
- `recycle`: after the change the product entry describes a different product.

**The label must come from evidence independent of how different the text looks.** If you label by reading the two
versions and judging how much changed, the calibration becomes circular and worthless. Acceptable evidence, roughly
from strongest to weakest:

1. A moderator or support action: the edit was reverted or the product was flagged or removed for this reason.
2. A change of identity in structured data at the same time: barcode, SKU, lab batch or product registration id,
   product type.
3. What happened to the menu items linked to the product: retailers unlinking them after the edit, or linked
   menu items that still plainly describe what the product used to be.
4. Customer or retailer evidence: reviews or tickets after the edit saying the product is not what it used to be.
5. A discontinuity around the edit in orders, price, or product photos.

For `edit` labels use positive evidence too where you can (identifiers unchanged, linked menu items still linked
and still matching, no complaints, no moderation action), not merely the absence of a recycle signal.

Record which evidence you used in `label_source` and how strong it is in `label_confidence` (`high`, `medium`,
`low`). If the evidence is absent or conflicting, set `label` to `"unsure"` in file 1 and leave the pair out of
file 2. Never guess, and never invent or synthesize pairs to fill a quota.

### Variant changes

Some edits are a matter of policy rather than fact: the same brand and category, but a different flavor, strain,
size, pack count, or potency. Set `variant_change` to one of `"flavor"`, `"strain"`, `"size"`, `"count"`,
`"potency"` for these (otherwise `null`). Label them only if a written policy or a moderator decision settles the
question; otherwise mark them `"unsure"`.

Decide `"size"` and `"potency"` from what the product is, not from the quantity columns alone: a potency that
moved from 21.3% to 22.8% is a routine batch update and not a variant change, while a 3.5 g product becoming a
28 g product is one. The quantities are the evidence the service will weigh for exactly these pairs, so a size or
potency variant whose quantity columns did not change (or were never filled in) is worth a line in the report.

**These are the most valuable pairs in the dataset.** Since every recycle stays within one brand, this is where
edits and recycles look most alike and where the threshold will actually sit. I want as many as you can find, and
I would rather have a hundred of them marked `unsure` than have them left out.

## Sampling

- Edits to brand products from the last 12 months. Only edits that changed the text of at least one descriptive
  field; skip edits to price, stock, or photos alone, and edits whose text is identical apart from whitespace.
  Count what you skip.
- Aim for at least 300 `edit` and 300 `recycle` pairs at `high` or `medium` confidence. Recycles are rare: take
  every well-evidenced one you can find, up to about 1,000, and tell me if you fall short.
- Spread across every top-level category roughly in proportion to edit volume, with a minimum of 20 pairs for any
  category that has them.
- One pair per product in files 1 and 2. Cap what any one brand contributes so that no brand supplies more than
  about 5% of the pairs of either label, and no single brand's writing style or product line dominates. If there
  are too few brands for that cap to leave enough pairs, relax it only as far as you must and tell me the cap you
  ended up with.
- In addition, a uniform random sample of 500 text edits to brand products, labeled as well as the evidence
  allows, tagged `"label_source": "random_sample"` plus the evidence used. This is what the base-rate estimate
  comes from. The per-brand cap does not apply to it.
- Assign `split`: 80% `"calibrate"`, 20% `"holdout"`, decided by a hash of the brand so that no brand appears in
  both. Check that the holdout ended up with enough recycles to measure anything (at least 40); if one large
  brand's hash tipped the balance, change the salt and say so.
  A third value, `"validation"`, is also understood: pairs that are never trained on and are used to choose
  between models, so that the holdout chooses nothing. It can be assigned here (for example 70/10/20, by the same
  brand hash) or carved out of `"calibrate"` afterwards with `record-similarity pairs carve`.

## Handling

- Read-only. Do not write to any production system.
- Keep the data inside this environment. Do not send product text to any external service, API, or tool.
- Replace product, brand, and retailer ids with a salted hash, stable across all files, and do not write the salt
  into any output. Brand names will still appear inside the text fields themselves; that is expected, and is one
  more reason the files stay here.
- No customer names, emails, or other personal data in any file; use review and ticket text as evidence for
  labels, but do not export it.
- If anything here conflicts with our data-handling policy, follow the policy and tell me what you changed.

## Before you finish

- Verify that every line of every `.jsonl` file parses and has exactly the keys specified, and that every line of
  `pairs.jsonl` meets the strict rules above.
- If the `record-similarity` binary and its model are available here, run
  `record-similarity calibrate --config config.example.toml --full pairs.full.jsonl` and paste its output into the
  report. It reads file 1 directly. Look at its last lines first: `record keys that were not compared` lists every
  key of a record that the service's schema does not define, with a count. A field you expected to be compared
  appearing there is a naming mismatch to fix or to tell me about. Then run it once more on file 2 without
  `--full`, which checks the strict format line by line.
- Finish with a short summary: what you produced, the counts, what fell short of the targets, and the decisions you
  need from me.
