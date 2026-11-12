//! Calibrating a match: catalog groups scored as `Match` would score them, with no server and no store.
//!
//! A catalog group is one stored product and the listings known to describe it (a brand's product and the menu
//! items linked to it). Scoring every listing against every product says how often the right product comes
//! first, and where a match threshold belongs.
//!
//! **The index holds every product; the split selects listings.** A listing has to be found among everything it
//! could be confused with, so `--split` narrows whose listings are scored and never which products they are
//! scored against.
//!
//! **Two rankings for each listing.** The *exhaustive* ranking scores every product exactly. The *served*
//! ranking is what `Match` would return: the products ordered by composite cosine, as a search orders them, the
//! first `depth * overfetch` of those scored exactly. Both use [`matching::rank`], the code the server runs. The
//! difference between the two is what the over-fetch loses, and it is measured, not argued.
//!
//! **The best other product stands in for "no product exists".** The groups hold only listings that have a
//! product. What a listing would be matched to were its own product absent is exactly the best-scoring product
//! that is not its own. A match threshold has to separate the two: at or below the scores listings give their own
//! products, and above the scores they give the best other.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use similarity_core::matching::{self, Candidate, Ranked};
use similarity_core::record::hex_prefix;
use similarity_core::vector::cosine;
use similarity_core::{Quantities, RecordVectors, Schema};

use super::pairs::{Ignored, data_lines, project};
use super::{Offline, RecordFields};
use crate::import::ImportLine;

/// One line of a groups file.
#[derive(Debug, Deserialize)]
struct Line {
  /// The group's name: the product's id in the index.
  group: String,
  #[serde(default)]
  owner: Option<String>,
  #[serde(default)]
  split: Option<String>,
  catalog: BTreeMap<String, Value>,
  listings: Vec<BTreeMap<String, Value>>,
}

/// A product in the index.
#[derive(Debug, Clone, PartialEq)]
pub struct Product {
  /// Its id: the group's name, or the id an export gave it.
  pub id: String,
  /// Who owns it, if the file says.
  pub owner: Option<String>,
  /// Its fields and quantities.
  pub fields: RecordFields,
}

/// A listing to be matched.
#[derive(Debug, Clone, PartialEq)]
pub struct Listing {
  /// A fingerprint of the group and the listing's content, so that two runs can be joined even if the file was
  /// reordered.
  pub id: String,
  /// The group it belongs to: the id of the product it is a listing of.
  pub group: String,
  /// The owner of that product.
  pub owner: Option<String>,
  /// The split its group is in.
  pub split: Option<String>,
  /// Its fields and quantities.
  pub fields: RecordFields,
}

/// What a set of groups was read from, and what was left out along the way.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GroupsInput {
  /// The SHA-256 of the groups file's text.
  pub sha256: String,
  /// The split whose listings are scored, if one was asked for.
  pub split: Option<String>,
  /// Record keys that were not compared, with how many values each lost.
  pub ignored: BTreeMap<(String, Ignored), usize>,
  /// Listings left out because their group is in another split.
  pub skipped_split: usize,
  /// Listings left out because the same group already had one with the same content.
  pub duplicate_listings: usize,
  /// Products that came from a catalog export, as distractors: they have no listings.
  pub distractors: usize,
}

/// The products and listings of one file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Groups {
  /// Every product, of every group and of the export: the index.
  pub products: Vec<Product>,
  /// The listings to score, in file order.
  pub listings: Vec<Listing>,
  /// Where they came from.
  pub input: GroupsInput,
}

fn listing_id(group: &str, fields: &RecordFields) -> anyhow::Result<String> {
  let bytes = serde_json::to_vec(&(group, fields)).context("a listing does not serialize")?;
  Ok(hex_prefix(&Sha256::digest(&bytes), 8))
}

/// Reads catalog groups, and optionally a catalog export whose records join the index as distractors.
///
/// Keys the schema does not define are counted and left out, as for pairs (D17). `split` selects whose listings
/// are scored; every product stays in the index.
///
/// # Errors
///
/// Returns an error naming the line if one is not a group, if a group has no listings, or if two products share
/// an id.
pub fn parse_groups(
  jsonl: &str,
  catalog: Option<&str>,
  schema: &Schema,
  split: Option<&str>,
) -> anyhow::Result<Groups> {
  let mut input = GroupsInput {
    sha256: hex_prefix(&Sha256::digest(jsonl.as_bytes()), 32),
    split: split.map(str::to_owned),
    ..GroupsInput::default()
  };
  let mut products = Vec::new();
  let mut listings = Vec::new();
  let mut seen: BTreeMap<String, String> = BTreeMap::new();
  for (number, line) in data_lines(jsonl) {
    let parsed: Line = serde_json::from_str(line).with_context(|| format!("line {number} is not a catalog group"))?;
    if let Some(first) = seen.insert(parsed.group.clone(), format!("line {number}")) {
      bail!(
        "{first} and line {number} are the same group (`{}`); a product is in the index once",
        parsed.group
      );
    }
    if parsed.listings.is_empty() {
      bail!("line {number}: the group `{}` has no listings", parsed.group);
    }
    let fields = project(parsed.catalog, schema, &mut input.ignored)
      .with_context(|| format!("line {number}: the product of `{}`", parsed.group))?;
    products.push(Product {
      id: parsed.group.clone(),
      owner: parsed.owner.clone(),
      fields,
    });

    let selected = split.is_none_or(|wanted| parsed.split.as_deref() == Some(wanted));
    let mut contents = BTreeSet::new();
    for (index, listing) in parsed.listings.into_iter().enumerate() {
      let fields = project(listing, schema, &mut input.ignored)
        .with_context(|| format!("line {number}: listing {index} of `{}`", parsed.group))?;
      if !selected {
        input.skipped_split = input.skipped_split.saturating_add(1);
        continue;
      }
      let id = listing_id(&parsed.group, &fields)?;
      if !contents.insert(id.clone()) {
        input.duplicate_listings = input.duplicate_listings.saturating_add(1);
        continue;
      }
      listings.push(Listing {
        id,
        group: parsed.group.clone(),
        owner: parsed.owner.clone(),
        split: parsed.split.clone(),
        fields,
      });
    }
  }

  for (number, line) in catalog.into_iter().flat_map(data_lines) {
    let parsed: ImportLine =
      serde_json::from_str(line).with_context(|| format!("line {number} of the catalog export is not a record"))?;
    if let Some(first) = seen.insert(parsed.id.clone(), format!("line {number} of the catalog export")) {
      bail!(
        "{first} and line {number} of the catalog export are both the product `{}`; a product is in the index once",
        parsed.id
      );
    }
    input.distractors = input.distractors.saturating_add(1);
    products.push(Product {
      id: parsed.id,
      owner: parsed.tenant,
      fields: parsed.record,
    });
  }
  Ok(Groups {
    products,
    listings,
    input,
  })
}

/// The product a listing would be matched to if its own were not in the catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct BestOther {
  /// How it scores against the listing.
  pub ranked: Ranked,
  /// Whether it belongs to the owner of the listing's own product: a sibling, which is the hard case.
  pub same_owner: bool,
}

/// One listing, scored against the whole index.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredListing {
  /// The listing.
  pub listing: Listing,
  /// Its own product, scored exactly against it.
  pub own: Ranked,
  /// Where its own product comes in the served ranking, from 1. `None` if it is not within the depth.
  pub own_rank: Option<usize>,
  /// Where its own product comes when every product is scored exactly, from 1.
  pub exhaustive_own_rank: usize,
  /// The best product that is not its own, among all of them. `None` in an index of one product.
  pub best_other: Option<BestOther>,
  /// Whether the served ranking's first `depth` are not the exhaustive ranking's: what the over-fetch lost.
  pub overfetch_lost: bool,
}

/// The result of scoring a groups file.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchCalibration {
  /// Every listing, scored, in file order.
  pub listings: Vec<ScoredListing>,
  /// How many products the index held.
  pub products: usize,
  /// How many candidates a served ranking returns.
  pub depth: usize,
  /// How many products are fetched for each one returned.
  pub overfetch: usize,
  /// What the groups were read from.
  pub input: GroupsInput,
}

struct Indexed {
  id: String,
  owner: Option<String>,
  vectors: RecordVectors,
  quantities: Quantities,
}

/// Scores every listing of `groups` against every product, as the server would serve it at `depth` candidates
/// and `overfetch`, and exhaustively.
///
/// # Errors
///
/// Returns an error if a record does not fit the schema or the model fails, or if a listing's own product is not
/// in the index (which parsing rules out).
pub fn calibrate_match(
  offline: &Offline,
  groups: Groups,
  depth: usize,
  overfetch: usize,
) -> anyhow::Result<MatchCalibration> {
  let schema = offline.schema();
  let dimension = offline.dimension();
  let mut index = Vec::with_capacity(groups.products.len());
  for product in &groups.products {
    let (vectors, quantities) = offline
      .prepared(&product.fields, None)
      .with_context(|| format!("the product `{}`", product.id))?;
    index.push(Indexed {
      id: product.id.clone(),
      owner: product.owner.clone(),
      vectors,
      quantities,
    });
  }
  let fetch = depth.saturating_mul(overfetch);

  let mut listings = Vec::with_capacity(groups.listings.len());
  for listing in groups.listings {
    let (vectors, quantities) = offline
      .prepared(&listing.fields, None)
      .with_context(|| format!("a listing of `{}`", listing.group))?;
    // The order a search would return the products in: by composite cosine, ties by id.
    let mut searched = Vec::with_capacity(index.len());
    for product in &index {
      searched.push((cosine(&vectors.composite, &product.vectors.composite)?, product));
    }
    searched.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
    let candidates = |take: usize| {
      searched.iter().take(take).map(|(similarity, product)| Candidate {
        id: &product.id,
        tenant: product.owner.as_deref(),
        search_similarity: *similarity,
        vectors: &product.vectors,
        quantities: &product.quantities,
      })
    };
    let served = matching::rank(schema, dimension, &vectors, &quantities, candidates(fetch), depth)?;
    let exhaustive = matching::rank(
      schema,
      dimension,
      &vectors,
      &quantities,
      candidates(usize::MAX),
      usize::MAX,
    )?;

    let position = |ranking: &[Ranked]| ranking.iter().position(|ranked| ranked.id == listing.group);
    let Some(exhaustive_at) = position(&exhaustive) else {
      bail!("the product of `{}` is not in the index", listing.group);
    };
    let same_ids = served
      .iter()
      .map(|ranked| ranked.id.as_str())
      .eq(exhaustive.iter().take(depth).map(|ranked| ranked.id.as_str()));
    let best_other = exhaustive
      .iter()
      .find(|ranked| ranked.id != listing.group)
      .map(|ranked| BestOther {
        ranked: ranked.clone(),
        same_owner: listing.owner.is_some() && ranked.tenant == listing.owner,
      });
    let Some(own) = exhaustive.get(exhaustive_at).cloned() else {
      bail!("the product of `{}` is not in the index", listing.group);
    };
    listings.push(ScoredListing {
      own,
      own_rank: position(&served).map(|at| at.saturating_add(1)),
      exhaustive_own_rank: exhaustive_at.saturating_add(1),
      best_other,
      overfetch_lost: !same_ids,
      listing,
    });
  }
  Ok(MatchCalibration {
    listings,
    products: index.len(),
    depth,
    overfetch,
    input: groups.input,
  })
}

#[cfg(test)]
pub(crate) mod testing {
  //! Match calibrations made by hand, for the tests of the statistics and the report: no model, no files.

  use similarity_core::score::Comparison;

  use super::*;

  /// A product scored `similarity` against a listing, with nothing else known about the comparison.
  pub(crate) fn ranked(id: &str, owner: Option<&str>, similarity: f32) -> Ranked {
    Ranked {
      id: id.to_owned(),
      tenant: owner.map(str::to_owned),
      search_similarity: similarity,
      comparison: Comparison {
        similarity,
        fields: Vec::new(),
      },
      quantities: Vec::new(),
    }
  }

  /// A listing of `group`, owned by `owner`, whose own product scores `own` and comes at `own_rank` in the served
  /// ranking, and whose best other product is `other` (an id, whether it is a sibling, and its score).
  pub(crate) fn scored(
    group: &str,
    owner: Option<&str>,
    own: f32,
    own_rank: Option<usize>,
    other: Option<(&str, bool, f32)>,
  ) -> ScoredListing {
    ScoredListing {
      listing: Listing {
        id: format!("{group}-{own}"),
        group: group.to_owned(),
        owner: owner.map(str::to_owned),
        split: None,
        fields: RecordFields::default(),
      },
      own: ranked(group, owner, own),
      own_rank,
      exhaustive_own_rank: own_rank.unwrap_or(99),
      best_other: other.map(|(id, same_owner, similarity)| BestOther {
        ranked: ranked(id, same_owner.then_some(owner).flatten(), similarity),
        same_owner,
      }),
      overfetch_lost: false,
    }
  }

  pub(crate) fn calibration(listings: Vec<ScoredListing>, depth: usize) -> MatchCalibration {
    MatchCalibration {
      listings,
      products: 3,
      depth,
      overfetch: 5,
      input: GroupsInput::default(),
    }
  }
}

#[cfg(test)]
mod tests {
  use similarity_core::testing::tiny_embedder;
  use similarity_core::{EmbedderOptions, FieldChange, FieldSpec, Policy, QuantityChange, QuantitySpec};

  use super::*;

  fn schema() -> Schema {
    let field = |name: &str, weight: f32| FieldSpec {
      name: name.to_owned(),
      weight,
      required: false,
      reject_below: None,
    };
    Schema::new(vec![field("name", 4.0), field("category", 2.0)])
      .unwrap()
      .with_quantities(vec![QuantitySpec {
        name: "weight".to_owned(),
        reject_above_factor: None,
        reject_incomparable: false,
      }])
      .unwrap()
  }

  const GROUPS: &str = r#"
    # Two brands, three products.
    {"group": "gummies", "owner": "brand-1", "split": "calibrate", "catalog": {"name": "blueberry gummies", "category": "edibles", "weight": {"value": 40, "unit": "g"}, "strain": "indica"}, "listings": [{"name": "wyld blueberry gummies mg"}, {"name": "blueberry gummies", "weight": {"value": 40, "unit": "g"}}, {"name": "wyld blueberry gummies mg"}]}
    {"group": "vape", "owner": "brand-1", "split": "calibrate", "catalog": {"name": "sativa vape cartridge", "category": "vape"}, "listings": [{"name": "vape cartridge sativa resin", "sku": 7}]}
    {"group": "flower", "owner": "brand-2", "split": "holdout", "catalog": {"name": "hybrid flower"}, "listings": [{"name": "flower hybrid indica", "category": "flower"}]}
  "#;

  fn parsed(split: Option<&str>) -> Groups {
    parse_groups(GROUPS, None, &schema(), split).unwrap()
  }

  #[test]
  fn the_split_selects_listings_and_the_index_keeps_every_product() {
    let all = parsed(None);
    let ids: Vec<&str> = all.products.iter().map(|product| product.id.as_str()).collect();
    assert_eq!(ids, ["gummies", "vape", "flower"]);
    assert_eq!(all.products[0].owner.as_deref(), Some("brand-1"));
    let groups: Vec<&str> = all.listings.iter().map(|listing| listing.group.as_str()).collect();
    assert_eq!(groups, ["gummies", "gummies", "vape", "flower"]);
    assert_eq!((all.input.skipped_split, all.input.split.as_deref()), (0, None));

    let holdout = parsed(Some("holdout"));
    assert_eq!(
      holdout.products, all.products,
      "a listing is scored against everything it could be confused with"
    );
    let groups: Vec<&str> = holdout.listings.iter().map(|listing| listing.group.as_str()).collect();
    assert_eq!(groups, ["flower"]);
    assert_eq!(
      (
        holdout.listings[0].owner.as_deref(),
        holdout.listings[0].split.as_deref()
      ),
      (Some("brand-2"), Some("holdout"))
    );
    assert_eq!(
      (holdout.input.skipped_split, holdout.input.split.as_deref()),
      (4, Some("holdout"))
    );
    assert!(parsed(Some("validation")).listings.is_empty());
    assert_eq!(
      holdout.input.sha256, all.input.sha256,
      "the hash is of the file, not of the selection"
    );
  }

  #[test]
  fn keys_the_schema_lacks_and_listings_given_twice_are_counted_and_left_out() {
    let all = parsed(None);
    assert_eq!(
      all.input.duplicate_listings, 1,
      "the gummies' first listing is there twice"
    );
    let ignored: Vec<(&str, Ignored, usize)> = all
      .input
      .ignored
      .iter()
      .map(|((key, reason), count)| (key.as_str(), *reason, *count))
      .collect();
    assert_eq!(
      ignored,
      [("sku", Ignored::NotInSchema, 1), ("strain", Ignored::NotInSchema, 1)]
    );
    assert!(all.products[0].fields.quantities.contains_key("weight"));
    assert!(!all.products[0].fields.texts.contains_key("strain"));
  }

  #[test]
  fn a_listing_keeps_its_id_when_the_file_is_reordered() {
    let reordered: String = GROUPS.lines().rev().collect::<Vec<_>>().join("\n");
    let ids = |groups: Groups| {
      let mut ids: Vec<(String, String)> = groups
        .listings
        .into_iter()
        .map(|listing| (listing.group, listing.id))
        .collect();
      ids.sort();
      ids
    };
    let original = ids(parsed(None));
    assert_eq!(ids(parse_groups(&reordered, None, &schema(), None).unwrap()), original);
    assert_eq!(original[0].1.len(), 16);
    let distinct: BTreeSet<&str> = original.iter().map(|(_, id)| id.as_str()).collect();
    assert_eq!(distinct.len(), original.len());
  }

  #[test]
  fn a_catalog_export_adds_products_to_the_index_and_no_listings() {
    let export = r#"
      {"id": "sku-9", "tenant": "brand-3", "record": {"name": "resin cartridge"}}
      {"id": "sku-10", "record": {"name": "indica flower"}, "anchor": {"name": "flower"}}
    "#;
    let with = parse_groups(GROUPS, Some(export), &schema(), None).unwrap();
    let ids: Vec<&str> = with.products.iter().map(|product| product.id.as_str()).collect();
    assert_eq!(ids, ["gummies", "vape", "flower", "sku-9", "sku-10"]);
    assert_eq!(with.products[3].owner.as_deref(), Some("brand-3"));
    assert_eq!(
      (with.input.distractors, with.listings.len()),
      (2, parsed(None).listings.len())
    );
  }

  #[test]
  fn groups_that_cannot_be_an_index_are_refused_with_the_line() {
    let schema = schema();
    let message =
      |jsonl: &str, export: Option<&str>| format!("{:#}", parse_groups(jsonl, export, &schema, None).unwrap_err());
    let twice = format!("{GROUPS}\n{}", GROUPS.lines().nth(2).unwrap());
    assert!(message(&twice, None).contains("the same group (`gummies`)"));
    let colliding = r#"{"id": "vape", "record": {"name": "x"}}"#;
    assert!(message(GROUPS, Some(colliding)).contains("both the product `vape`"));
    let empty = r#"{"group": "g", "catalog": {"name": "x"}, "listings": []}"#;
    assert!(message(empty, None).contains("has no listings"));
    assert!(message("{\"group\": \"g\"}", None).contains("line 1 is not a catalog group"));
    assert!(message(GROUPS, Some("not json")).contains("line 1 of the catalog export"));
    let bad_quantity =
      r#"{"group": "g", "catalog": {"name": "x"}, "listings": [{"name": "y", "weight": {"unit": "g"}}]}"#;
    assert!(message(bad_quantity, None).contains("listing 0 of `g`"));
  }

  fn offline() -> Offline {
    let options = EmbedderOptions {
      max_tokens: 16,
      batch_size: 8,
      output_dim: None,
    };
    Offline::new(tiny_embedder(&options), schema(), Policy::new(0.5, None).unwrap())
  }

  /// The whole of it on a tiny random model: whatever the scores, every listing is ranked against every product,
  /// its own product is found, and what is recorded about it is what the rankings say.
  #[test]
  fn every_listing_is_ranked_against_the_whole_index_as_a_match_would_rank_it() {
    let offline = offline();
    let calibration = calibrate_match(&offline, parsed(None), 2, 1).unwrap();
    assert_eq!(
      (calibration.products, calibration.depth, calibration.overfetch),
      (3, 2, 1)
    );
    assert_eq!(calibration.listings.len(), 4);
    for scored in &calibration.listings {
      assert_eq!(scored.own.id, scored.listing.group);
      assert!((1..=3).contains(&scored.exhaustive_own_rank));
      let other = scored.best_other.as_ref().unwrap();
      assert_ne!(other.ranked.id, scored.listing.group);
      // The best other is ahead of the own product exactly when the own product is not first.
      assert_eq!(
        other.ranked.comparison.similarity > scored.own.comparison.similarity,
        scored.exhaustive_own_rank > 1,
        "{scored:?}"
      );
      // Two candidates served from two fetched: the own product is served only if the search had it in its
      // first two, and then at no better a rank than scoring everything gives it.
      if let Some(rank) = scored.own_rank {
        assert!(rank <= 2 && rank <= scored.exhaustive_own_rank);
      }
      assert!(scored.own.comparison.fields.iter().all(|field| !field.below_floor));
    }

    // The listing that repeats its product's name and weight: unchanged on both, and first.
    let exact = &calibration.listings[1];
    let name = exact
      .own
      .comparison
      .fields
      .iter()
      .find(|field| field.field == "name")
      .unwrap();
    assert_eq!((name.change, name.similarity), (FieldChange::Unchanged, Some(1.0)));
    assert_eq!(exact.own.comparison.similarity, 1.0);
    assert_eq!(exact.own.quantities[0].change, QuantityChange::Unchanged);
    // The model here is random, and made anew for every run: which product comes second is not fixed, and no
    // assertion may depend on it. Whichever it is, it is a sibling exactly when it has the listing's owner. The
    // gummies and the vape are one brand's; the flower is the only product of the other.
    let owner_of = |group: &str| if group == "flower" { "brand-2" } else { "brand-1" };
    for scored in &calibration.listings {
      let other = scored.best_other.as_ref().unwrap();
      assert_eq!(
        other.same_owner,
        owner_of(&other.ranked.id) == owner_of(&scored.listing.group),
        "{scored:?}"
      );
    }
    assert_eq!(calibration.listings[3].listing.group, "flower");
    assert!(!calibration.listings[3].best_other.as_ref().unwrap().same_owner);

    // Fetching everything serves the exhaustive ranking: nothing is lost.
    let wide = calibrate_match(&offline, parsed(None), 3, 5).unwrap();
    for scored in &wide.listings {
      assert_eq!(scored.own_rank, Some(scored.exhaustive_own_rank));
      assert!(!scored.overfetch_lost);
    }
    // One candidate from one fetched is the search's favorite, which is the own product or is not.
    let narrow = calibrate_match(&offline, parsed(None), 1, 1).unwrap();
    for (scored, full) in narrow.listings.iter().zip(&wide.listings) {
      assert_eq!(scored.exhaustive_own_rank, full.exhaustive_own_rank);
      assert_eq!(
        scored.own, full.own,
        "the exact score does not depend on what was served"
      );
      if scored.own_rank.is_none() && full.exhaustive_own_rank == 1 {
        assert!(scored.overfetch_lost, "the right product was first and was not fetched");
      }
    }
  }

  #[test]
  fn an_index_of_one_product_has_no_best_other() {
    let single = r#"{"group": "only", "catalog": {"name": "blueberry gummies"}, "listings": [{"name": "gummies"}]}"#;
    let groups = parse_groups(single, None, &schema(), None).unwrap();
    let calibration = calibrate_match(&offline(), groups, 5, 5).unwrap();
    let scored = &calibration.listings[0];
    assert_eq!((scored.own_rank, scored.exhaustive_own_rank), (Some(1), 1));
    assert!(scored.best_other.is_none());
  }
}
