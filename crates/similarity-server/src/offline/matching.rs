//! Calibrating a match: catalog groups scored as `Match` would score them, with no server and no store.
//!
//! A catalog group is one stored product and the menu items known to describe it (a brand's product and the menu
//! items linked to it). Scoring every menu item against every product says how often the right product comes
//! first, and where a match threshold belongs.
//!
//! **The index holds every product; the split selects menu items.** A menu item has to be found among everything it
//! could be confused with, so `--split` narrows whose menu items are scored and never which products they are
//! scored against.
//!
//! **Two rankings for each menu item.** The *exhaustive* ranking scores every product exactly. The *served*
//! ranking is what `Match` would return: the products ordered by composite cosine, as a search orders them, the
//! first `depth * overfetch` of those scored exactly. Both use [`matching::rank`], the code the server runs. The
//! difference between the two is what the over-fetch loses, and it is measured, not argued.
//!
//! **The best other product stands in for "no product exists".** The groups hold only menu items that have a
//! product. What a menu item would be matched to were its own product absent is exactly the best-scoring product
//! that is not its own. A match threshold has to separate the two: at or below the scores menu items give their own
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

use super::pairs::{Ignored, data_lines, listings_are_now_menu_items, owner_is_now_brand_id, project};
use super::{Offline, RecordFields};
use crate::import::ImportLine;

/// One line of a groups file.
#[derive(Debug, Deserialize)]
struct Line {
  /// The group's name: the product's id in the index.
  group: String,
  #[serde(default, rename = "owner", deserialize_with = "owner_is_now_brand_id")]
  _owner: (),
  #[serde(default, rename = "listings", deserialize_with = "listings_are_now_menu_items")]
  _listings: (),
  #[serde(default)]
  brand_id: Option<String>,
  #[serde(default)]
  split: Option<String>,
  catalog: BTreeMap<String, Value>,
  menu_items: Vec<BTreeMap<String, Value>>,
}

/// A product in the index.
#[derive(Debug, Clone, PartialEq)]
pub struct Product {
  /// Its id: the group's name, or the id an export gave it.
  pub id: String,
  /// Who owns it, if the file says.
  pub brand_id: Option<String>,
  /// Its fields and quantities.
  pub fields: RecordFields,
}

/// A menu item to be matched.
#[derive(Debug, Clone, PartialEq)]
pub struct MenuItem {
  /// A fingerprint of the group and the menu item's content, so that two runs can be joined even if the file was
  /// reordered.
  pub id: String,
  /// The group it belongs to: the id of the product it is a menu item for.
  pub group: String,
  /// The brand of that product.
  pub brand_id: Option<String>,
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
  /// The split whose menu items are scored, if one was asked for.
  pub split: Option<String>,
  /// Record keys that were not compared, with how many values each lost.
  pub ignored: BTreeMap<(String, Ignored), usize>,
  /// Menu items left out because their group is in another split.
  pub skipped_split: usize,
  /// Menu items left out because the same group already had one with the same content.
  pub duplicate_menu_items: usize,
  /// Products that came from a catalog export, as distractors: they have no menu items.
  pub distractors: usize,
}

/// The products and menu items of one file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Groups {
  /// Every product, of every group and of the export: the index.
  pub products: Vec<Product>,
  /// The menu items to score, in file order.
  pub menu_items: Vec<MenuItem>,
  /// Where they came from.
  pub input: GroupsInput,
}

fn menu_item_id(group: &str, fields: &RecordFields) -> anyhow::Result<String> {
  let bytes = serde_json::to_vec(&(group, fields)).context("a menu item does not serialize")?;
  Ok(hex_prefix(&Sha256::digest(&bytes), 8))
}

/// Reads catalog groups, and optionally a catalog export whose records join the index as distractors.
///
/// Keys the schema does not define are counted and left out, as for pairs (D17). `split` selects whose menu items
/// are scored; every product stays in the index.
///
/// # Errors
///
/// Returns an error naming the line if one is not a group, if a group has no menu items, or if two products share
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
  let mut menu_items = Vec::new();
  let mut seen: BTreeMap<String, String> = BTreeMap::new();
  for (number, line) in data_lines(jsonl) {
    let parsed: Line = serde_json::from_str(line).with_context(|| format!("line {number} is not a catalog group"))?;
    if let Some(first) = seen.insert(parsed.group.clone(), format!("line {number}")) {
      bail!(
        "{first} and line {number} are the same group (`{}`); a product is in the index once",
        parsed.group
      );
    }
    if parsed.menu_items.is_empty() {
      bail!("line {number}: the group `{}` has no menu items", parsed.group);
    }
    let fields = project(parsed.catalog, schema, &mut input.ignored)
      .with_context(|| format!("line {number}: the product of `{}`", parsed.group))?;
    products.push(Product {
      id: parsed.group.clone(),
      brand_id: parsed.brand_id.clone(),
      fields,
    });

    let selected = split.is_none_or(|wanted| parsed.split.as_deref() == Some(wanted));
    let mut contents = BTreeSet::new();
    for (index, menu_item) in parsed.menu_items.into_iter().enumerate() {
      let fields = project(menu_item, schema, &mut input.ignored)
        .with_context(|| format!("line {number}: menu item {index} of `{}`", parsed.group))?;
      if !selected {
        input.skipped_split = input.skipped_split.saturating_add(1);
        continue;
      }
      let id = menu_item_id(&parsed.group, &fields)?;
      if !contents.insert(id.clone()) {
        input.duplicate_menu_items = input.duplicate_menu_items.saturating_add(1);
        continue;
      }
      menu_items.push(MenuItem {
        id,
        group: parsed.group.clone(),
        brand_id: parsed.brand_id.clone(),
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
      brand_id: parsed.brand_id,
      fields: parsed.record,
    });
  }
  Ok(Groups {
    products,
    menu_items,
    input,
  })
}

/// The product a menu item would be matched to if its own were not in the catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct BestOther {
  /// How it scores against the menu item.
  pub ranked: Ranked,
  /// Whether it belongs to the brand of the menu item's own product: a sibling, which is the hard case.
  pub same_brand: bool,
}

/// One menu item, scored against the whole index.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredMenuItem {
  /// The menu item.
  pub menu_item: MenuItem,
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
  /// Every menu item, scored, in file order.
  pub menu_items: Vec<ScoredMenuItem>,
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
  brand_id: Option<String>,
  vectors: RecordVectors,
  quantities: Quantities,
}

/// Scores every menu item of `groups` against every product, as the server would serve it at `depth` candidates
/// and `overfetch`, and exhaustively.
///
/// # Errors
///
/// Returns an error if a record does not fit the schema or the model fails, or if a menu item's own product is not
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
      brand_id: product.brand_id.clone(),
      vectors,
      quantities,
    });
  }
  let fetch = depth.saturating_mul(overfetch);

  let mut menu_items = Vec::with_capacity(groups.menu_items.len());
  for menu_item in groups.menu_items {
    let (vectors, quantities) = offline
      .prepared(&menu_item.fields, None)
      .with_context(|| format!("a menu item of `{}`", menu_item.group))?;
    // The order a search would return the products in: by composite cosine, ties by id.
    let mut searched = Vec::with_capacity(index.len());
    for product in &index {
      searched.push((cosine(&vectors.composite, &product.vectors.composite)?, product));
    }
    searched.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
    let candidates = |take: usize| {
      searched.iter().take(take).map(|(similarity, product)| Candidate {
        id: &product.id,
        brand_id: product.brand_id.as_deref(),
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

    let position = |ranking: &[Ranked]| ranking.iter().position(|ranked| ranked.id == menu_item.group);
    let Some(exhaustive_at) = position(&exhaustive) else {
      bail!("the product of `{}` is not in the index", menu_item.group);
    };
    let same_ids = served
      .iter()
      .map(|ranked| ranked.id.as_str())
      .eq(exhaustive.iter().take(depth).map(|ranked| ranked.id.as_str()));
    let best_other = exhaustive
      .iter()
      .find(|ranked| ranked.id != menu_item.group)
      .map(|ranked| BestOther {
        ranked: ranked.clone(),
        same_brand: menu_item.brand_id.is_some() && ranked.brand_id == menu_item.brand_id,
      });
    let Some(own) = exhaustive.get(exhaustive_at).cloned() else {
      bail!("the product of `{}` is not in the index", menu_item.group);
    };
    menu_items.push(ScoredMenuItem {
      own,
      own_rank: position(&served).map(|at| at.saturating_add(1)),
      exhaustive_own_rank: exhaustive_at.saturating_add(1),
      best_other,
      overfetch_lost: !same_ids,
      menu_item,
    });
  }
  Ok(MatchCalibration {
    menu_items,
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

  /// A product scored `similarity` against a menu item, with nothing else known about the comparison.
  pub(crate) fn ranked(id: &str, brand_id: Option<&str>, similarity: f32) -> Ranked {
    Ranked {
      id: id.to_owned(),
      brand_id: brand_id.map(str::to_owned),
      search_similarity: similarity,
      comparison: Comparison {
        similarity,
        fields: Vec::new(),
      },
      quantities: Vec::new(),
    }
  }

  /// A menu item of `group`, owned by `brand_id`, whose own product scores `own` and comes at `own_rank` in the served
  /// ranking, and whose best other product is `other` (an id, whether it is a sibling, and its score).
  pub(crate) fn scored(
    group: &str,
    brand_id: Option<&str>,
    own: f32,
    own_rank: Option<usize>,
    other: Option<(&str, bool, f32)>,
  ) -> ScoredMenuItem {
    ScoredMenuItem {
      menu_item: MenuItem {
        id: format!("{group}-{own}"),
        group: group.to_owned(),
        brand_id: brand_id.map(str::to_owned),
        split: None,
        fields: RecordFields::default(),
      },
      own: ranked(group, brand_id, own),
      own_rank,
      exhaustive_own_rank: own_rank.unwrap_or(99),
      best_other: other.map(|(id, same_brand, similarity)| BestOther {
        ranked: ranked(id, same_brand.then_some(brand_id).flatten(), similarity),
        same_brand,
      }),
      overfetch_lost: false,
    }
  }

  pub(crate) fn calibration(menu_items: Vec<ScoredMenuItem>, depth: usize) -> MatchCalibration {
    MatchCalibration {
      menu_items,
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
    {"group": "gummies", "brand_id": "brand-1", "split": "calibrate", "catalog": {"name": "blueberry gummies", "category": "edibles", "weight": {"value": 40, "unit": "g"}, "strain": "indica"}, "menu_items": [{"name": "wyld blueberry gummies mg"}, {"name": "blueberry gummies", "weight": {"value": 40, "unit": "g"}}, {"name": "wyld blueberry gummies mg"}]}
    {"group": "vape", "brand_id": "brand-1", "split": "calibrate", "catalog": {"name": "sativa vape cartridge", "category": "vape"}, "menu_items": [{"name": "vape cartridge sativa resin", "sku": 7}]}
    {"group": "flower", "brand_id": "brand-2", "split": "holdout", "catalog": {"name": "hybrid flower"}, "menu_items": [{"name": "flower hybrid indica", "category": "flower"}]}
  "#;

  fn parsed(split: Option<&str>) -> Groups {
    parse_groups(GROUPS, None, &schema(), split).unwrap()
  }

  /// Until 2026-11-12 a group's brand was its `owner` and its menu items were its `listings`. A file from before
  /// is refused by name: read as unknown keys, the first would cost every product its brand without a word.
  #[test]
  fn a_groups_file_under_the_names_from_before_is_refused_and_told_the_new_ones() {
    let renamed = |from: &str, to: &str| {
      let old = GROUPS.replace(&format!("\"{to}\":"), &format!("\"{from}\":"));
      assert_ne!(old, GROUPS, "the fixture has a `{to}` to turn back into `{from}`");
      format!("{:#}", parse_groups(&old, None, &schema(), None).unwrap_err())
    };
    let error = renamed("owner", "brand_id");
    assert!(error.contains("`owner` is now `brand_id`"), "{error}");
    let error = renamed("listings", "menu_items");
    assert!(error.contains("`listings` is now `menu_items`"), "{error}");
  }

  #[test]
  fn the_split_selects_menu_items_and_the_index_keeps_every_product() {
    let all = parsed(None);
    let ids: Vec<&str> = all.products.iter().map(|product| product.id.as_str()).collect();
    assert_eq!(ids, ["gummies", "vape", "flower"]);
    assert_eq!(all.products[0].brand_id.as_deref(), Some("brand-1"));
    let groups: Vec<&str> = all
      .menu_items
      .iter()
      .map(|menu_item| menu_item.group.as_str())
      .collect();
    assert_eq!(groups, ["gummies", "gummies", "vape", "flower"]);
    assert_eq!((all.input.skipped_split, all.input.split.as_deref()), (0, None));

    let holdout = parsed(Some("holdout"));
    assert_eq!(
      holdout.products, all.products,
      "a menu item is scored against everything it could be confused with"
    );
    let groups: Vec<&str> = holdout
      .menu_items
      .iter()
      .map(|menu_item| menu_item.group.as_str())
      .collect();
    assert_eq!(groups, ["flower"]);
    assert_eq!(
      (
        holdout.menu_items[0].brand_id.as_deref(),
        holdout.menu_items[0].split.as_deref()
      ),
      (Some("brand-2"), Some("holdout"))
    );
    assert_eq!(
      (holdout.input.skipped_split, holdout.input.split.as_deref()),
      (4, Some("holdout"))
    );
    assert!(parsed(Some("validation")).menu_items.is_empty());
    assert_eq!(
      holdout.input.sha256, all.input.sha256,
      "the hash is of the file, not of the selection"
    );
  }

  #[test]
  fn keys_the_schema_lacks_and_menu_items_given_twice_are_counted_and_left_out() {
    let all = parsed(None);
    assert_eq!(
      all.input.duplicate_menu_items, 1,
      "the gummies' first menu item is there twice"
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
  fn a_menu_item_keeps_its_id_when_the_file_is_reordered() {
    let reordered: String = GROUPS.lines().rev().collect::<Vec<_>>().join("\n");
    let ids = |groups: Groups| {
      let mut ids: Vec<(String, String)> = groups
        .menu_items
        .into_iter()
        .map(|menu_item| (menu_item.group, menu_item.id))
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
  fn a_catalog_export_adds_products_to_the_index_and_no_menu_items() {
    let export = r#"
      {"id": "sku-9", "brand_id": "brand-3", "record": {"name": "resin cartridge"}}
      {"id": "sku-10", "record": {"name": "indica flower"}, "anchor": {"name": "flower"}}
    "#;
    let with = parse_groups(GROUPS, Some(export), &schema(), None).unwrap();
    let ids: Vec<&str> = with.products.iter().map(|product| product.id.as_str()).collect();
    assert_eq!(ids, ["gummies", "vape", "flower", "sku-9", "sku-10"]);
    assert_eq!(with.products[3].brand_id.as_deref(), Some("brand-3"));
    assert_eq!(
      (with.input.distractors, with.menu_items.len()),
      (2, parsed(None).menu_items.len())
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
    let empty = r#"{"group": "g", "catalog": {"name": "x"}, "menu_items": []}"#;
    assert!(message(empty, None).contains("has no menu items"));
    assert!(message("{\"group\": \"g\"}", None).contains("line 1 is not a catalog group"));
    assert!(message(GROUPS, Some("not json")).contains("line 1 of the catalog export"));
    let bad_quantity =
      r#"{"group": "g", "catalog": {"name": "x"}, "menu_items": [{"name": "y", "weight": {"unit": "g"}}]}"#;
    assert!(message(bad_quantity, None).contains("menu item 0 of `g`"));
  }

  fn offline() -> Offline {
    let options = EmbedderOptions {
      max_tokens: 16,
      batch_size: 8,
      output_dim: None,
    };
    Offline::new(tiny_embedder(&options), schema(), Policy::new(0.5, None).unwrap())
  }

  /// The whole of it on a tiny random model: whatever the scores, every menu item is ranked against every product,
  /// its own product is found, and what is recorded about it is what the rankings say.
  #[test]
  fn every_menu_item_is_ranked_against_the_whole_index_as_a_match_would_rank_it() {
    let offline = offline();
    let calibration = calibrate_match(&offline, parsed(None), 2, 1).unwrap();
    assert_eq!(
      (calibration.products, calibration.depth, calibration.overfetch),
      (3, 2, 1)
    );
    assert_eq!(calibration.menu_items.len(), 4);
    for scored in &calibration.menu_items {
      assert_eq!(scored.own.id, scored.menu_item.group);
      assert!((1..=3).contains(&scored.exhaustive_own_rank));
      let other = scored.best_other.as_ref().unwrap();
      assert_ne!(other.ranked.id, scored.menu_item.group);
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

    // The menu item that repeats its product's name and weight: unchanged on both, and first.
    let exact = &calibration.menu_items[1];
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
    // assertion may depend on it. Whichever it is, it is a sibling exactly when it has the menu item's brand. The
    // gummies and the vape are one brand's; the flower is the only product of the other.
    let brand_of = |group: &str| if group == "flower" { "brand-2" } else { "brand-1" };
    for scored in &calibration.menu_items {
      let other = scored.best_other.as_ref().unwrap();
      assert_eq!(
        other.same_brand,
        brand_of(&other.ranked.id) == brand_of(&scored.menu_item.group),
        "{scored:?}"
      );
    }
    assert_eq!(calibration.menu_items[3].menu_item.group, "flower");
    assert!(!calibration.menu_items[3].best_other.as_ref().unwrap().same_brand);

    // Fetching everything serves the exhaustive ranking: nothing is lost.
    let wide = calibrate_match(&offline, parsed(None), 3, 5).unwrap();
    for scored in &wide.menu_items {
      assert_eq!(scored.own_rank, Some(scored.exhaustive_own_rank));
      assert!(!scored.overfetch_lost);
    }
    // One candidate from one fetched is the search's favorite, which is the own product or is not.
    let narrow = calibrate_match(&offline, parsed(None), 1, 1).unwrap();
    for (scored, full) in narrow.menu_items.iter().zip(&wide.menu_items) {
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
    let single = r#"{"group": "only", "catalog": {"name": "blueberry gummies"}, "menu_items": [{"name": "gummies"}]}"#;
    let groups = parse_groups(single, None, &schema(), None).unwrap();
    let calibration = calibrate_match(&offline(), groups, 5, 5).unwrap();
    let scored = &calibration.menu_items[0];
    assert_eq!((scored.own_rank, scored.exhaustive_own_rank), (Some(1), 1));
    assert!(scored.best_other.is_none());
  }
}
