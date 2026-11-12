# Plan: menu-item to brand-product matching (feature 20)

**Built (D32).** What differs from this plan: the trainer already left out groups by owner (D24), and now also
by a group's own `split`; the two populations are written as a `Calibration` so the existing statistics apply;
`Describe` and the client report the match settings; the request log line is written by `grpc.rs` like the
others; and there is no `MAX_MATCH_FETCH` setting, only the constant.

A plan for answering a second question with the same model and the same stored catalog: not "is this edit still the
same brand product?" but "which brand product is this retailer's menu item a listing of?". It was written before
any of it was built, against `initial-implementation` at `545ef20`. Read `CLAUDE.md` first, then
`docs/decisions.md` D5, D7, D12, D14, D16, and D17, which this plan leans on throughout.

## Contents

- [What it is for](#what-it-is-for)
- [What already exists](#what-already-exists)
- [What the suggestion got wrong or left open](#what-the-suggestion-got-wrong-or-left-open)
- [Design](#design)
- [The work, in order](#the-work-in-order)
- [Tests](#tests)
- [Documentation to update](#documentation-to-update)
- [Dependencies on other features](#dependencies-on-other-features)
- [What can only be validated at work](#what-can-only-be-validated-at-work)
- [Decisions for the owner](#decisions-for-the-owner)
- [Size](#size)
- [Out of scope](#out-of-scope)

## What it is for

Retailers list brand products on their menus in their own words: "WYLD Blueberry 100mg (10pk)" for the brand's
"Blueberry Gummies 100mg". A menu item often has only a name. Linking a menu item to the brand product it is a
listing of lets a marketplace show the brand's content, aggregate listings, and spot a menu item that no longer
describes what it is linked to. The service already holds every brand product's vectors in Qdrant, embedded field
by field; what is missing is a request that scores a menu item against that catalog *correctly* when the two
sides have different fields filled in, a decision with a measured threshold, and the offline measurement that
produces the threshold.

## What already exists

Verified against the code at `545ef20`:

| Piece | Where | What it gives this feature |
|---|---|---|
| `FindSimilar` with an ad hoc record | `similarity.proto` (`FindSimilarRequest.record`), `grpc.rs` `find_similar`, `engine.rs` `Engine::find_similar` | Embeds an unstored record under the product schema and searches the collection, optionally within one tenant. This is already a working, if skewed, matcher. |
| Search across tenants | `SearchQuery.tenant` is optional (`similarity-store/src/lib.rs`); `QdrantStore::search` filters on layout always and tenant only when given | A match across all brands needs no new filter. |
| The exact weighted score from two composites | `score::compare` (per-field detail, `Unchanged` from equal text hashes), `score::weighted_similarity`, `Comparison::weigh` (`similarity-core/src/score.rs`) | The re-rank. A field one side lacks is reported and left out of the mean: exactly the fix D5 names. |
| Block slicing | `Schema::blocks` (`schema.rs`) | Per-field cosines from any two composites of one layout, under any weights. |
| Quantity comparison | `score::compare_quantities`, `quantity::compare` (`quantity.rs`) | With the product as baseline, no anchor, and the menu item as candidate: a factor, `incomparable`, `zero_crossing` per quantity. |
| Catalog groups | `similarity-train/src/data.rs` `GroupLine { catalog, listings }`, `group_examples`; the in-batch ranking loss (D13) | The trainer already fine-tunes for this task. `docs/data-request-prompt.md` section 4 asks for `catalog-groups.jsonl`. |
| A threshold between two score populations | `offline::calibrate::suggest_threshold`, `offline::stats::bootstrap` (owner resampling, private today), `operating_points` | The calibration statistics, once generalized from `Label` to two populations. |
| Flat result events | `offline::events`, D17 | The shape `calibrate-match --ndjson` follows. |
| A catalog export format | `import::ImportLine` (`import.rs`), D20 | Distractor products for the offline evaluation, read with the existing parser. |

## What the suggestion got wrong or left open

1. **"Optionally narrowed by a brand-name hint" cannot be a filter.** The server stores no text (D9), and the
   payload holds no brand name (`PointPayload` in `qdrant.rs`: id, tenant, layout, hashes, revision, timestamp,
   quantities). The tenant is the brand's *key*, as a pseudonym. Two hints already work with no new parameter: a
   caller that knows the brand's key passes it as `tenant` (an exact, indexed filter), and a caller that has only
   the brand's name puts it in the menu item's `brand` field, where it is scored like any field. Nothing new is
   needed; the plan adds no hint parameter.
2. **"Over-fetches five times" is a guess, and the skew it fixes can be stated exactly.** For a query `Q` and a
   stored record `C` (sets of filled fields, `W` the sum of their weights), the cosine of the two composites is
   `exact * W(Q∩C) / sqrt(W(Q) * W(C))`. The factor depends only on which fields the candidate has filled. If every
   product has the same fields filled, the search order *is* the exact order and over-fetching gains nothing; the
   more uneven the catalog, the more it gains. For a name-only query with the example weights the factor runs from
   1 (a product with only a name) down to `sqrt(4 / 10) = 0.63` (a product with all four fields). So the factor is
   a cost setting, kept in the config with the suggestion's 5 as its default, and `calibrate-match` measures what
   it loses against an exhaustive ranking on the real catalog.
3. **The re-rank needs the candidates' vectors, which `search` does not return.** `Neighbor` is id, tenant,
   similarity. Re-ranking 50 candidates with 50 `get` calls is 50 round trips. The store trait grows one method
   that returns the vectors and quantities with each hit, so the contract tests and both stores grow (the
   suggestion's "Fit" did not list the store crate).
4. **"Its own field configuration" would break the search.** A different field set is a different layout (D12):
   a menu-item composite could not be compared with a product composite at all. Menu items must be mapped onto the
   product schema's field names by the caller (as products already are, README "Mapping your records"), and a
   field with no counterpart is left out. Only the *weights* could differ without a new layout, because blocks are
   recovered from any composite whatever it was weighted with (D5). The first version uses the schema's weights;
   separate match weights are an owner decision, and the per-field events make them measurable without
   re-embedding.
5. **`catalog-groups.jsonl` is not an evaluation set as the data request defines it.** Section 4 asks for groups
   to be used as *training* data and to leave out every brand in the `holdout` split. Calibrating a match threshold
   on the groups the model was trained on would overstate it. The groups need a `split` key, assigned by the same
   brand hash as the pairs, and the holdout brands' groups have to be exported instead of dropped. The trainer's
   `GroupLine` has no `split` and trains on every group today, so it grows the same split filter pairs have.
6. **Groups contain only menu items that have a product.** In real traffic some menu items have none in the
   catalog, and a matcher that always matches something will link them to the nearest product. The groups can
   still measure that case: for each listing the best-scoring product *other than its own* is exactly what it
   would be matched to were its product absent. The base rate of unmatched menu items cannot be measured from
   groups, so the data request has to ask for it.
7. **Recall and MRR measure ranking; they do not pick a threshold.** The threshold separates two populations per
   listing: the score of its own product and the score of the best other product. That is the edit/recycle
   separation of D17 in another guise, and it should carry the same owner-resampled intervals and operating
   points.
8. **Quantities were left undecided** (`docs/plan-features.md`). They are a strong signal for a match (a 1 g and
   a 3.5 g listing of one strain are, for most catalogs, different brand products) and the reason D16 keeps them
   out of the text score applies unchanged. They are reported per candidate and do not move the score or the
   decision in the first version. Also: `quantity::compare` measures a *change*, as a factor; a product's THC range
   and a menu item's single batch value are better compared by containment. That is noted, not built.
9. **New RPC or mode of `FindSimilar` was not settled.** See the design: a new RPC.

## Design

### The shape of it

The smallest useful version is one read-only RPC, `Match`, on the existing service, against the existing brand
product collection, plus one offline command, `calibrate-match`. No new collection, no new layout, no new model,
no stored menu items, no new dependency.

`Match` and not a mode of `FindSimilar`, because the two answer different questions with different contracts:
`FindSimilar` documents its similarity as the composite cosine and returns ids; `Match` returns the exact score
over the fields both sides have, per-field and per-quantity detail, and a decision under its own threshold.
Changing `FindSimilar`'s similarity would silently change a documented number for existing callers. Both share
one engine path for embedding and searching.

`Match` never writes. D7, D8, and D20 are untouched by construction, and D14 does not arise: the tenant rule
governs requests that name a stored id, and a match names none. Searching across tenants is what `FindSimilar`
already allows. It remains a consistency check and not access control.

### Core: `crates/similarity-core/src/matching.rs` (new, synchronous, no I/O)

Shared by the engine and the offline command, so that the server and the calibration rank identically.

```rust
/// One stored product as a search returned it, before re-ranking.
pub struct Candidate<'a> {
  pub id: &'a str,
  pub tenant: Option<&'a str>,
  /// The composite cosine the search ranked it by.
  pub search_similarity: f32,
  pub vectors: &'a RecordVectors,
  pub quantities: &'a Quantities,
}

/// A candidate scored exactly against the listing.
pub struct Ranked {
  pub id: String,
  pub tenant: Option<String>,
  pub search_similarity: f32,
  /// `score::compare` with the product as baseline, no anchor, and the listing as candidate, weighed under the
  /// schema: fields only one side has are reported and left out of `comparison.similarity`.
  pub comparison: Comparison,
  /// `score::compare_quantities` the same way round. Never consulted by `similarity` or the decision.
  pub quantities: Vec<QuantityScore>,
}

pub enum Decision { NotJudged, Matched, NoMatch }

/// Scores every candidate, sorts by exact similarity (ties by id), and keeps `limit`.
///
/// # Errors
/// [`Error::Incompatible`] if a composite does not have this schema's layout.
pub fn rank(schema: &Schema, dimension: usize, listing: &RecordVectors, listing_quantities: &Quantities,
  candidates: impl IntoIterator<Item = Candidate<'_>>, limit: usize) -> Result<Vec<Ranked>>;

/// `NotJudged` without a threshold; otherwise whether the top candidate reaches it.
pub fn decide(threshold: Option<f32>, ranked: &[Ranked]) -> Decision;
```

Floors (`reject_below`) are revision rules and do not apply: `rank` clears `below_floor` (weighs under a copy of
the schema with no floors) so a match response never reports one.

### Store: `crates/similarity-store/src/lib.rs`

```rust
/// A search hit with what a re-rank needs.
pub struct Found {
  pub neighbor: Neighbor,
  /// The hit's current vectors and text fingerprints.
  pub vectors: RecordVectors,
  pub quantities: Quantities,
}

/// `search`, returning each hit's current vectors and quantities. Same filters, order, and limit as `search`.
///
/// # Errors
/// As [`VectorStore::search`], and [`StoreError::Corrupt`] if a hit cannot be decoded.
fn search_vectors(&self, query: SearchQuery<'_>) -> impl Future<Output = Result<Vec<Found>>> + Send;
```

- `MemoryStore`: the brute-force loop it has, cloning the record's `vectors` and `quantities`.
- `QdrantStore`: the same `QueryPointsBuilder` as `search`, with `.with_vectors(vec![CURRENT])` (only the `current`
  vector; the anchor stays on disk, D6) and the payload, decoded with the existing `decode_payload` and the vector
  decoding `get` already does. `qdrant-client` 1.20 implements `From<Vec<&str>> for WithVectorsSelector`
  (`grpc_conversions/primitives.rs`).
- A response of 50 candidates carries 50 composites of 1,024 floats, about 200 KB. Fine; feature 12 would measure
  it under load.

### Config: `[matching]` in `crates/similarity-server/src/config.rs`

`match` is a Rust keyword, so the table is `matching`:

```toml
[matching]
# Match the top candidate only at or above this exact similarity. A measurement, like `policy.threshold`: read it
# from `record-similarity calibrate-match` on held-out catalog groups. Without it, `Match` ranks and never decides.
# threshold = ...

# How many candidates to fetch per one returned, before re-ranking exactly. A cost setting, not a threshold:
# `calibrate-match` reports what it loses against an exhaustive ranking.
overfetch = 5
```

`threshold` is optional with no default and the example config ships it commented out (D3, D4). `overfetch` is an
integer of at least 1, defaulting to 5. The number fetched is `limit * overfetch`, capped at a constant
`MAX_MATCH_FETCH = 1000`. The table is optional; `deny_unknown_fields` stays on.

### Engine: `crates/similarity-server/src/engine.rs`

```rust
pub struct MatchInput { pub listing: RecordInput, pub limit: usize, pub tenant: Option<String>,
  pub threshold: Option<f32> }
pub struct MatchOutput { pub candidates: Vec<Ranked>, pub decision: Decision, pub threshold: Option<f32>,
  pub fetched: usize }
pub async fn match_listing(&self, input: MatchInput) -> Result<MatchOutput, EngineError>;
```

- A listing with an `id` or a `tenant` is `InvalidArgument`: a menu item's id and its retailer belong to another
  system, and accepting them would invite a caller to confuse the retailer with the brand tenant. The search scope
  is `MatchInput.tenant`.
- The listing is prepared under the product schema (`Engine::prepare`), so an unknown field or quantity is refused
  and `name`, being required, must be present, exactly as for `FindSimilar`.
- `limit`: zero means 5, at most `MAX_NEIGHBORS` (100). A per-request `threshold` overrides the configured one and
  must lie in `[-1, 1]`.
- Embeds with `self.embed(&prepared, None)`, calls `search_vectors`, then `matching::rank` and `matching::decide`.
- Logs one `match` line through `tracing`: `candidates`, `fetched`, `top_similarity` (rounded, D15), `decision`.
- `Description` gains the matching settings.

### API: `similarity.proto` (additive)

```proto
// Finds the stored records a listing from elsewhere describes (a retailer's menu item, say), best first, and
// decides whether the best is the same product. Read-only: nothing is stored or changed.
//
// Unlike `FindSimilar`, each candidate is scored exactly over the fields both it and the listing have, so a
// listing with only a name is scored on its name whatever else a stored record has filled in. Quantities are
// compared and reported per candidate and never move a similarity or the decision.
//
// Errors: `INVALID_ARGUMENT` for a listing that does not fit the schema, that carries an `id` or a `tenant`, or a
// `limit` above 100; `RESOURCE_EXHAUSTED` and `UNAVAILABLE` as for `Evaluate`.
rpc Match(MatchRequest) returns (MatchResponse);

message MatchRequest {
  // Field names and quantities as the server's schema defines them (see `Describe`); `id` and `tenant` must be
  // absent. Map the listing's own fields onto these names; leave out a field with no counterpart.
  Record listing = 1;
  // How many candidates to return. Defaults to 5; at most 100.
  uint32 limit = 2;
  // Only consider records of this tenant (the brand, when the caller knows it).
  optional string tenant = 3;
  // Overrides the server's match threshold for this request. Cosine scale, [-1, 1].
  optional float threshold = 4;
}

enum MatchDecision {
  MATCH_DECISION_UNSPECIFIED = 0;
  // No match threshold is configured or given: candidates are ranked, not judged.
  MATCH_DECISION_NOT_JUDGED = 1;
  // The first candidate reaches the threshold.
  MATCH_DECISION_MATCHED = 2;
  // No candidate reaches it, or there were none.
  MATCH_DECISION_NO_MATCH = 3;
}

message MatchCandidate {
  string id = 1;
  optional string tenant = 2;
  // The weighted mean of per-field similarities over the fields both sides have.
  float similarity = 3;
  // The score the search ranked it by (what `FindSimilar` would report).
  float search_similarity = 4;
  // Read with the stored record as the baseline and the listing as the record: `FIELD_CHANGE_ADDED` is a field
  // only the listing has, `FIELD_CHANGE_REMOVED` one only the stored record has. Never `RESTORED`, never
  // `below_floor`.
  repeated FieldScore fields = 5;
  // The same reading for quantities. Never `beyond_limit`.
  repeated QuantityScore quantities = 6;
}

message MatchResponse {
  repeated MatchCandidate candidates = 1;
  MatchDecision decision = 2;
  // The threshold that was applied, if any.
  optional float threshold = 3;
}
```

`DescribeResponse` gains `optional float match_threshold = 9` and `uint32 match_overfetch = 10`. `grpc.rs` maps
types only, reusing the existing `FieldScore` and `QuantityScore` conversions. The example client gains
`match <listing.json> [--tenant T] [--limit N]`.

### Offline: `record-similarity calibrate-match`

```text
record-similarity calibrate-match --config config.toml catalog-groups.jsonl
  [--catalog export.jsonl] [--split holdout] [--depth 20] [--ndjson]
```

- **Input.** Each group line: `group`, `owner`, optional `split`, `catalog` (the product), `listings`. Fields and
  quantities read with `RecordFields`, as `calibrate --full` reads pairs, with keys the schema does not define
  counted and reported (D17). `--catalog` adds products from an import export (`ImportLine`) as distractors; they
  have no listings. Every product, of every group and of the export, is in the index; `--split` selects only which
  groups' *listings* are scored. A product id is `group` for group products and the export's `id` otherwise; a
  collision is refused.
- **Listing id.** A hash of the group and the listing's content, as a pair without `pair_id` gets (D17), so runs
  can be joined even if the file is reordered. Two identical listings in one group are counted once and reported.
- **Scoring.** Every product and listing is embedded with `Offline`'s embedder. For each listing, the composite
  cosine to every product gives the search order; `matching::rank` over the first `depth * overfetch` of those
  gives the served ranking, and over all of them the exhaustive ranking. Same code as the server; the store is
  replaced by brute force, which is what `MemoryStore` does anyway. Cost grows with listings times products and has
  to be measured on the real catalog; the default index (the groups' own products, about 2,000) is small.
- **Statistics,** all in `offline::stats` (CLAUDE.md), with owner-resampled 95% intervals (D17):
  recall at 1, 5, and `depth`; mean reciprocal rank (zero beyond `depth`); the range of own-product scores and of
  best-other scores, overall and per field; the suggested threshold separating the two (`suggest_threshold`,
  generalized from `Label` to two populations); operating points ("to match a listing with no product wrongly at
  most 1%, 2%, 5% of the time, how many listings with a product go unmatched?"); the wrong-match rate when the
  product is present (best other ahead of own, and at or above the threshold); the share of best-other products
  from the same owner (sibling products, the hard case); listings whose top `depth` differs between served and
  exhaustive ranking (the over-fetch loss); and, per quantity, how often the own product's quantity is
  incomparable or changed by a factor above 1. `stats::bootstrap` becomes `pub(crate)` for this.
- **Text report** for a person, opening with the fingerprint and width (D11). **Events** with `--ndjson`, through
  `similarity-log` as results, flat, joined on `listing_id` (D17), carrying `schema_field` and `schema_quantity`
  lines as calibration events do:
  - `match_listing`: `listing_id`, `group`, `owner`, `split`, `own_rank` (absent beyond `depth`),
    `own_similarity`, `best_other_similarity`, `best_other_same_owner`, `margin` (own minus best other),
    `exhaustive_own_rank`, `overfetch_lost`, and, with a threshold configured, `matched` and `correct`.
  - `match_field`: `listing_id`, `field`, `presence` (`both`, `listing`, `product`), `own_similarity`,
    `best_other_similarity`. Enough to judge other weights or a field floor again without a model, like
    `calibrate-diff --config-b`.
  - `match_quantity`: `listing_id`, `quantity`, `change`, `factor`, `incomparable`, `zero_crossing`, against the
    own product.
- `just calibrate-match groups config *flags`, and a synthetic `data/catalog-groups.sample.jsonl` (twenty groups or
  so, three flavors of one gummy line among them, marked synthetic as the pair samples are). No threshold is ever
  read from it.

### Trainer: `crates/similarity-train/src/data.rs`

`GroupLine` gains `split: Option<String>`, and `group_examples` takes a `Split` like `pair_examples`, so groups in
`--holdout-split` are never trained on and a file without splits is all training data, as for pairs. The data
fingerprint changes with what is selected, so `RunIdentity` covers it without a new field. The parity tests are
unaffected (data selection only).

### Data request: `docs/data-request-prompt.md` section 4

- Add `split` to every group, by the same brand hash as file 1, and export the holdout brands' groups (marked
  `holdout`) instead of leaving them out.
- Ask for the fraction of menu items, in a uniform sample, that are linked to no brand product, and whether that
  is because none exists or because nobody linked it.
- Ask how a menu item's fields map onto `name`, `category`, `brand`, `description` and the four quantities, and
  whether menu-item quantities are per item or per package.
- Ask whether a different size or pack of one product is a different brand product (decides whether a quantity
  should ever reject a match).
- Optionally, the brand product catalog as an import export, as distractors.

## The work, in order

Each step leaves `just ci` green, tests written first where practical.

1. **Data request and trainer split.** The section 4 changes above; `GroupLine.split` and the `Split` filter in
   `group_examples`, wired to the existing `--split` and `--holdout-split`. Run `just test-parity` once (it should
   not move). This step is useful alone: it can go to work before anything else.
2. **Core `matching` module.** `Candidate`, `Ranked`, `Decision`, `rank`, `decide`, exported from `lib.rs`, with
   example tests and the properties below. Run `PROPTEST_CASES=5000 just test` (it touches scoring).
3. **Store `search_vectors`.** Trait, `MemoryStore`, `QdrantStore`, contract tests. `just test-qdrant`.
4. **Engine and config.** `[matching]`, `MatchInput`, `MatchOutput`, `Engine::match_listing`, `Description`; the
   state machine in `similarity-server/src/properties.rs` gains `Match` requests.
5. **Proto, `grpc.rs`, example client.** One round trip in `lib.rs`; one ignored real-model test in
   `tests/real_e2e.rs`. `just test-real`.
6. **`calibrate-match`.** `offline/matching.rs` (input, scoring), additions to `offline/stats.rs`, rendering in
   `offline/report.rs`, events in `offline/events.rs`, the subcommand in `main.rs`, the `justfile` recipe, the
   synthetic sample. Run it on the sample and put the output in the README only as an example of the format.
7. **Documentation.** Below. D22.

Steps 2 and 3 are independent of each other; 1 is independent of everything.

## Tests

Fast unless marked. Names are the test functions.

**`similarity-core`, `matching`:**

- `a_name_only_listing_scores_its_name_similarity_against_a_fully_filled_product`
- `a_field_only_one_side_has_is_reported_and_left_out_of_the_match_score`
- `identical_text_is_reported_unchanged_at_similarity_one`
- `candidates_are_ordered_by_exact_score_not_by_search_score`
- `equal_match_scores_are_ordered_by_id`
- `no_threshold_means_nothing_is_judged`
- `the_top_candidate_is_matched_at_the_threshold_and_not_below_it`
- `an_empty_candidate_list_is_no_match`
- `quantities_are_reported_per_candidate_and_never_move_its_score`
- `a_match_never_reports_a_field_below_its_floor`

**`similarity-core`, `properties`:**

- `the_search_score_is_the_exact_score_scaled_by_the_field_overlap` (the formula in "what the suggestion got
  wrong", item 2, on generated records with random field subsets)
- `ranking_every_stored_record_finds_the_exact_top_candidates`
- `a_listing_whose_texts_all_appear_in_a_product_scores_one_against_it`
- `ranking_is_independent_of_candidate_order`

**`similarity-store`, `tests/contract.rs`** (each for both stores; the Qdrant ones `#[ignore]`d with
`just test-qdrant`):

- `a_vector_search_returns_each_hits_stored_vectors_and_quantities`
- `a_vector_search_applies_the_same_filters_order_and_limit_as_search`

**`similarity-server`, engine:**

- `a_name_only_listing_finds_its_product_ahead_of_a_better_filled_wrong_one` (the D5 case: with `WordBagEmbedder`,
  a wrong product with every field filled outranks the right one in search and is put back behind it by the
  re-rank)
- `matching_never_changes_what_is_stored`
- `a_listing_with_an_id_or_a_tenant_is_refused`
- `a_match_can_be_limited_to_one_tenant`
- `a_match_limit_above_the_maximum_is_refused`
- `a_listing_with_an_unknown_field_is_refused`
- `a_match_against_an_empty_store_returns_no_candidates_and_no_match`
- `a_per_request_threshold_outside_the_cosine_range_is_refused`

**`similarity-server`, `properties`:** the state machine generates `Match` requests and checks after each that the
store is exactly what it was.

**`similarity-server`, config:** `the_example_config_sets_no_match_threshold`, `an_overfetch_of_zero_is_refused`.

**`similarity-server`, gRPC (`lib.rs`):** `a_match_round_trips_over_grpc`.

**`similarity-server`, `tests/real_e2e.rs`** (`#[ignore]`, `just test-real`, needs the model):
`the_real_model_ranks_a_menu_style_name_first_against_its_own_product`, on the three example records.

**`similarity-server`, offline:**

- `a_listing_whose_product_ranks_first_counts_toward_recall_at_one`
- `the_best_other_score_is_the_highest_product_that_is_not_the_listings_own`
- `a_product_beyond_the_depth_adds_zero_to_mean_reciprocal_rank`
- `a_listing_id_survives_reordering_the_file`
- `the_split_selects_listings_and_the_index_keeps_every_product`
- `a_catalog_export_adds_distractors_and_no_listings`
- `a_product_id_used_twice_is_refused`
- `overfetch_loss_is_counted_when_the_served_and_exhaustive_rankings_differ`
- `match_events_are_flat_and_joined_on_listing_id`
- `the_match_report_and_the_events_give_the_same_numbers`
- `unknown_keys_in_a_group_are_counted_and_reported`

**`similarity-train`, data:** `catalog_groups_in_the_holdout_split_are_not_trained_on`,
`a_catalog_group_without_a_split_is_training_data`.

## Documentation to update

- **`docs/decisions.md`**: a new **D22, "Matching a listing to a product is a read, scored exactly"**: why a new
  RPC and not a mode; the skew formula and why over-fetch is a cost setting measured by `calibrate-match`; why the
  menu item is mapped onto the product schema instead of having its own (D12); why `Match` never writes and
  refuses an id or tenant; quantities reported and kept out (D16); the two populations a match threshold
  separates, and why the best other product stands in for "no product exists". D5's last paragraph points to it.
- **`README.md`**: the RPC table in "Making requests" and a grpcurl example; a section "Matching menu items to
  products" under "Integrating it" (mapping a menu item's fields, the two brand hints, what the response means); a
  subsection "Choosing a match threshold" under "Choosing thresholds" with real output of `just calibrate-match` on
  the synthetic sample; `[matching]` in the configuration reference; "Fine-tuning" on the groups' `split`; the
  "Search scores drop" limitation rewritten to say `Match` scores exactly and `FindSimilar` does not; a new
  limitation that quantities are reported, not used, in a match. Every number run, not derived (CLAUDE.md).
- **`CLAUDE.md`**: under "easy to get wrong": `matching.threshold` is a measurement and the example ships none;
  `Match` never writes; a listing is mapped onto the product schema and never gets a layout of its own;
  `calibrate-match` event names are an interface (D17 extended).
- **`docs/feature-suggestions.md`** and **`docs/plan-features.md`**: mark 20 as planned, then built.
- **`config.example.toml`**: the `[matching]` table, threshold commented out.

## Dependencies on other features

None is required. Interfaces assumed if they arrive first or later:

- **19 (reason codes on the wire):** a match decision has no reasons beyond "below threshold"; if 19 adds a
  `ReasonDetail` message, `MatchResponse` can carry one later. Nothing here waits for it.
- **8 (calibration diff):** a `calibrate-match` diff is not built; the flat events, `match_field` especially, are
  shaped so that one can join two runs on `listing_id` later.
- **12 (latency and load):** the cost of `Match` (fetching `limit * overfetch` composites) should be measured
  there; no figure for it is quoted until then.
- **17 (reviewer feedback):** reviewers confirming or correcting a match would produce exactly the groups this
  needs. Not assumed.

## What can only be validated at work

Everything that says whether this works: recall and MRR on held-out groups, the match threshold, the over-fetch
loss on the real catalog's mix of filled fields, how often sibling products beat the right one, how quantities
behave on true matches, whether fine-tuning on the training groups helps the held-out ones, and the base rate of
menu items with no product. Here everything is built and tested on synthetic data and `WordBagEmbedder`; the
sample's numbers demonstrate the format only.

## Decisions for the owner

| Decision | Recommended default |
|---|---|
| A new `Match` RPC, or a mode of `FindSimilar`? | A new RPC. `FindSimilar` keeps its documented score. |
| Should `FindSimilar` get the exact re-rank too? | No, not now. It can take a flag later using the same code. |
| Search scope when no tenant is given | Every tenant. `tenant` narrows to one brand when the caller knows it. |
| A brand-name hint parameter | None. The brand's key as `tenant`, or its name in the `brand` field. |
| Separate match weights | Not in the first version; the schema's weights. Add `[matching.weights]` only if `match_field` events show a gain on real groups. |
| Field floors for matching | None. Measured later from `match_field` if wanted. |
| Decision rule | Threshold on the top score only. `margin` is reported so a margin rule can be judged later. |
| Quantities in a match | Reported per candidate, never in the score or the decision. Whether a size difference should reject waits for the data request's answer on what a brand product is. |
| Groups' `split` | Add it, export the holdout brands' groups, and have the trainer skip `holdout` groups. |
| A listing carrying `id` or `tenant` | Refused. |
| Store menu items | No. Matching is read-only. Storing them (to catch recycled menu items) would be a separate collection and feature. |
| Over-fetch default | 5, as suggested, revisited from `calibrate-match`'s over-fetch loss on the real catalog. |

## Size

**Large.** The server half (core module, one store method, engine, config, proto) is medium on its own and has no
open technical risk. What makes it large is the offline command with its statistics, events, and report held to
D17's standards, the change to the data request and the trainer, the documentation that every number in it be
measured, and that its value can only be judged at work, after a new export with held-out groups.

## Out of scope

- A second collection or schema for menu items, and storing them.
- Using quantities to re-rank or decide; containment of a value in a product's range.
- A `calibrate-match` diff, or a match variant of `calibrate-diff --config-b`.
- Changing `FindSimilar`.
- Any change to the encoder, the thresholds for `Evaluate`, or the shipped model.
