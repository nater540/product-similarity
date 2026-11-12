//! Training examples, and the order they are fed to the model in.
//!
//! Two kinds of signal, matching the two files the data-request prompt (`docs/data-request-prompt.md`) asks for:
//!
//! - **Labeled pairs** (`pairs.full.jsonl` or `pairs.jsonl`): the before and after of one edit, labeled `edit` or
//!   `recycle`. Each field whose text changed contributes one [`PairExample`]: the two texts should embed close
//!   together for an edit and far apart for a recycle.
//! - **Catalog groups** (`catalog-groups.jsonl`): merchant listings linked to the same catalog product. Each listing
//!   contributes one [`GroupExample`] per field: its text should embed closer to its own catalog entry than to any
//!   other catalog entry in the batch.
//!
//! Everything here is plain data and deterministic given a seed; no tensors are involved.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;
use similarity_core::record::normalize_text;
// The generator that orders examples. Re-exported because it has always been part of this module's interface.
pub use similarity_core::rng::Rng;

use crate::error::{Result, TrainError};

/// Two versions of one field's text, and whether they describe the same product.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairExample {
  /// The field's text before the edit.
  pub before: String,
  /// The field's text after the edit.
  pub after: String,
  /// `true` for an ordinary edit, `false` for a recycled listing.
  pub same: bool,
}

/// One field of a catalog product and of a merchant listing linked to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupExample {
  /// The catalog entry's text.
  pub catalog: String,
  /// The merchant listing's text for the same field.
  pub listing: String,
}

/// What a pairs file yielded, and what was left out of it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PairExamples {
  /// The usable examples.
  pub examples: Vec<PairExample>,
  /// Pairs whose label is neither `edit` nor `recycle` (`unsure`, typically).
  pub skipped_label: usize,
  /// Pairs assigned to a different split (the hold-out, typically).
  pub skipped_split: usize,
}

impl PairExamples {
  /// How many examples come from ordinary edits.
  pub fn edits(&self) -> usize {
    self.examples.iter().filter(|example| example.same).count()
  }

  /// How many examples come from recycled listings.
  pub fn recycles(&self) -> usize {
    self.examples.len().saturating_sub(self.edits())
  }
}

// Unknown keys are ignored on purpose: `pairs.full.jsonl` carries a dozen bookkeeping keys per line, and extra
// fields per record, that only matter to whoever assembled it.
#[derive(Deserialize)]
struct PairLine {
  label: String,
  #[serde(default)]
  split: Option<String>,
  #[serde(default)]
  owner: Option<String>,
  before: BTreeMap<String, Value>,
  after: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct GroupLine {
  #[serde(default)]
  owner: Option<String>,
  #[serde(default)]
  split: Option<String>,
  catalog: BTreeMap<String, Value>,
  listings: Vec<BTreeMap<String, Value>>,
}

/// A field's text in the form the server embeds it: whitespace collapsed, `None` if absent, blank, or not a string.
///
/// "Not a string" is what keeps a record's quantities out of training. They are objects (`{"value": 3.5, "unit":
/// "g"}`), the server compares them as numbers and never embeds them (docs/decisions.md, D16), and so there is
/// nothing about them for the encoder to learn.
fn field_text(record: &BTreeMap<String, Value>, field: &str) -> Option<String> {
  record.get(field).and_then(Value::as_str).and_then(normalize_text)
}

/// Parses a JSONL document, skipping blank lines and `#` comments, and naming the line of anything malformed.
fn parse_lines<T: serde::de::DeserializeOwned>(jsonl: &str, origin: &Path) -> Result<Vec<T>> {
  jsonl
    .lines()
    .enumerate()
    .filter(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
    .map(|(index, line)| {
      serde_json::from_str(line).map_err(|source| TrainError::Data {
        path: origin.to_path_buf(),
        line: index.saturating_add(1),
        reason: source.to_string(),
      })
    })
    .collect()
}

/// Reads a file, attaching its path to a failure.
///
/// # Errors
///
/// Returns [`TrainError::Io`] if the file cannot be read as UTF-8 text.
pub fn read(path: &Path) -> Result<String> {
  std::fs::read_to_string(path).map_err(|source| TrainError::Io {
    path: path.to_path_buf(),
    source,
  })
}

/// Which pairs of a file to use, by their `split` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split<'a> {
  /// Pairs assigned to this split, and pairs assigned to none (a file without splits is all training data).
  Training(&'a str),
  /// Only pairs explicitly assigned to this split. A pair with no `split` is never held out by accident.
  HeldOut(&'a str),
}

impl Split<'_> {
  fn admits(self, assigned: Option<&str>) -> bool {
    match (self, assigned) {
      (Self::Training(wanted) | Self::HeldOut(wanted), Some(assigned)) => assigned == wanted,
      (Self::Training(_), None) => true,
      (Self::HeldOut(_), None) => false,
    }
  }
}

/// Turns labeled before/after pairs into per-field examples.
///
/// Only pairs labeled `edit` or `recycle` are used, and only those `split` admits: the hold-out must stay unseen
/// during training, or calibrating on it afterwards measures nothing. A field contributes an example only if both
/// sides have a value and the values differ: an unchanged field teaches nothing, and a field only one side has
/// cannot be compared.
///
/// # Errors
///
/// Returns [`TrainError::Data`] naming the line if a line is not a JSON object with `label`, `before`, and `after`.
pub fn pair_examples(jsonl: &str, origin: &Path, fields: &[String], split: Split<'_>) -> Result<PairExamples> {
  let mut found = PairExamples::default();
  for line in parse_lines::<PairLine>(jsonl, origin)? {
    let same = match line.label.as_str() {
      "edit" => true,
      "recycle" => false,
      _ => {
        found.skipped_label = found.skipped_label.saturating_add(1);
        continue;
      }
    };
    if !split.admits(line.split.as_deref()) {
      found.skipped_split = found.skipped_split.saturating_add(1);
      continue;
    }
    for field in fields {
      if let (Some(before), Some(after)) = (field_text(&line.before, field), field_text(&line.after, field))
        && before != after
      {
        found.examples.push(PairExample { before, after, same });
      }
    }
  }
  Ok(found)
}

/// The owners of every pair assigned to one of `splits`.
///
/// These are the brands a model is to be measured on, so nothing of theirs may be trained on: not their pairs,
/// which the split already keeps out, and not their catalog groups, which have no split of their own.
///
/// # Errors
///
/// Returns [`TrainError::Data`] naming the line if a line is not a JSON object with `label`, `before`, and `after`.
pub fn owners_in(jsonl: &str, origin: &Path, splits: &[&str]) -> Result<BTreeSet<String>> {
  Ok(
    parse_lines::<PairLine>(jsonl, origin)?
      .into_iter()
      .filter(|line| line.split.as_deref().is_some_and(|split| splits.contains(&split)))
      .filter_map(|line| line.owner)
      .collect(),
  )
}

/// What a catalog-groups file yielded, and what was left out of it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GroupExamples {
  /// The usable examples.
  pub examples: Vec<GroupExample>,
  /// Groups left out because their owner is one a model is measured on, or because they are themselves in a
  /// split that is.
  pub excluded_groups: usize,
  /// Groups that name no owner. They are trained on, since there is nothing to check them against.
  pub groups_without_owner: usize,
}

/// Turns catalog groups into per-field (catalog text, listing text) examples.
///
/// A listing whose text is identical to the catalog's is left out: it is already at similarity 1. A group whose
/// `owner` is in `exclude` is left out whole: a brand whose pairs are held back for measuring must not be learned
/// from through its catalog either, or the measurement is of something the model has seen. So is a group whose
/// own `split` is one of `held_out`: those are the groups a match threshold is measured on (`calibrate-match`).
/// A group with no `split` is training data, as a pair with none is.
///
/// # Errors
///
/// Returns [`TrainError::Data`] naming the line if a line is not a JSON object with `catalog` and `listings`.
pub fn group_examples(
  jsonl: &str,
  origin: &Path,
  fields: &[String],
  exclude: &BTreeSet<String>,
  held_out: &[&str],
) -> Result<GroupExamples> {
  let mut found = GroupExamples::default();
  for line in parse_lines::<GroupLine>(jsonl, origin)? {
    if line.split.as_deref().is_some_and(|split| held_out.contains(&split)) {
      found.excluded_groups = found.excluded_groups.saturating_add(1);
      continue;
    }
    match &line.owner {
      Some(owner) if exclude.contains(owner) => {
        found.excluded_groups = found.excluded_groups.saturating_add(1);
        continue;
      }
      Some(_) => {}
      None => found.groups_without_owner = found.groups_without_owner.saturating_add(1),
    }
    let examples = &mut found.examples;
    for field in fields {
      let Some(catalog) = field_text(&line.catalog, field) else {
        continue;
      };
      for listing in &line.listings {
        if let Some(listing) = field_text(listing, field)
          && listing != catalog
        {
          examples.push(GroupExample {
            catalog: catalog.clone(),
            listing,
          });
        }
      }
    }
  }
  Ok(found)
}

/// One training step's worth of examples, as indexes into the example lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Batch {
  /// Indexes of [`PairExample`]s, trained with the contrastive loss.
  Pairs(Vec<usize>),
  /// Indexes of [`GroupExample`]s, trained with the ranking loss.
  Groups(Vec<usize>),
}

impl Batch {
  /// How many examples the batch holds.
  pub fn len(&self) -> usize {
    match self {
      Self::Pairs(indexes) | Self::Groups(indexes) => indexes.len(),
    }
  }

  /// Whether the batch holds no examples.
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

/// Splits `order` into batches of at most `batch_size` in which no text appears twice.
///
/// `texts` gives the two texts of each example. The ranking loss treats every other example in the batch as a
/// negative, so a batch holding the same catalog entry twice would be told that two listings of one product must
/// embed apart, which is exactly the opposite of what the data says. An example that collides with one already in
/// the batch is deferred to a later batch rather than dropped, so every example is used exactly once per pass.
fn batches_without_duplicates(texts: &[(&str, &str)], order: Vec<usize>, batch_size: usize) -> Vec<Vec<usize>> {
  let batch_size = batch_size.max(1);
  let mut batches = Vec::new();
  let mut remaining = order;
  while !remaining.is_empty() {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut batch = Vec::with_capacity(batch_size);
    let mut deferred = Vec::new();
    for index in remaining {
      let Some((first, second)) = texts.get(index) else {
        continue;
      };
      // The first candidate always fits (nothing has been seen yet), so every pass makes progress.
      if batch.len() < batch_size && !seen.contains(first) && !seen.contains(second) {
        seen.insert(first);
        seen.insert(second);
        batch.push(index);
      } else {
        deferred.push(index);
      }
    }
    if !batch.is_empty() {
      batches.push(batch);
    }
    remaining = deferred;
  }
  batches
}

/// The batches of one pass over all the data, in the order they are trained on.
///
/// Each batch comes from a single kind of example, because the two kinds use different losses. The batches of both
/// kinds are then shuffled together, so the model sees them interleaved in proportion to how much of each there is
/// rather than all of one and then all of the other.
///
/// Only catalog groups are kept free of repeated texts. The contrastive loss scores each pair on its own, so a
/// text appearing in two pairs of a batch is harmless, and labeled pairs repeat texts by nature (the same listing
/// as the "before" of an edit and of a recycle).
///
/// A catalog batch that would hold a single example is left out of the pass. With nothing to rank it against, its
/// loss and gradient are exactly zero, so it teaches nothing; but an optimizer step on it would still move the
/// weights along whatever momentum the earlier steps left, and would use up a step of the learning-rate schedule.
/// Such batches appear when one catalog entry has far more listings than the rest. The order is reshuffled every
/// pass, so over several passes different listings of that entry are the ones left out.
pub fn epoch(pairs: &[PairExample], groups: &[GroupExample], batch_size: usize, rng: &mut Rng) -> Vec<Batch> {
  let mut shuffled = |count: usize| {
    let mut order: Vec<usize> = (0..count).collect();
    rng.shuffle(&mut order);
    order
  };
  let pair_order = shuffled(pairs.len());
  let group_order = shuffled(groups.len());

  let group_texts: Vec<(&str, &str)> = groups
    .iter()
    .map(|example| (example.catalog.as_str(), example.listing.as_str()))
    .collect();

  let mut batches: Vec<Batch> = pair_order
    .chunks(batch_size.max(1))
    .map(|chunk| Batch::Pairs(chunk.to_vec()))
    .chain(
      batches_without_duplicates(&group_texts, group_order, batch_size)
        .into_iter()
        .filter(|batch| batch.len() >= 2)
        .map(Batch::Groups),
    )
    .collect();
  rng.shuffle(&mut batches);
  batches
}

#[cfg(test)]
mod tests {
  use super::*;

  fn fields(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
  }

  fn origin() -> &'static Path {
    Path::new("pairs.jsonl")
  }

  #[test]
  fn each_changed_field_of_a_labeled_pair_becomes_an_example() {
    let jsonl = [
      "# a comment, then a blank line",
      "",
      concat!(
        r#"{"label": "edit", "before": {"name": "Blueberry  Gummies", "description": "Ten pieces."}, "#,
        r#""after": {"name": "Blueberry Gummies 100mg", "description": "Ten pieces, vegan."}}"#
      ),
    ]
    .join("\n");
    let found = pair_examples(
      &jsonl,
      origin(),
      &fields(&["name", "description"]),
      Split::Training("calibrate"),
    )
    .unwrap();
    assert_eq!(found.examples.len(), 2);
    assert_eq!(
      found.examples[0],
      PairExample {
        before: "Blueberry Gummies".to_owned(),
        after: "Blueberry Gummies 100mg".to_owned(),
        same: true
      }
    );
    assert_eq!((found.edits(), found.recycles()), (2, 0));
  }

  #[test]
  fn unchanged_one_sided_and_unrequested_fields_are_left_out() {
    let jsonl = concat!(
      r#"{"label": "recycle", "before": {"name": "Gummies", "brand": "Wyld", "description": "x", "strain_id": 7}, "#,
      r#""after": {"name": "Gummies ", "brand": "Kiva", "category": "Vapes", "description": "y"}}"#
    );
    let found = pair_examples(
      jsonl,
      origin(),
      &fields(&["name", "description", "category"]),
      Split::Training("calibrate"),
    )
    .unwrap();
    // `name` only differs in whitespace, `category` is one-sided, `brand` was not asked for.
    assert_eq!(
      found.examples,
      [PairExample {
        before: "x".to_owned(),
        after: "y".to_owned(),
        same: false
      }]
    );
    assert_eq!((found.edits(), found.recycles()), (0, 1));
  }

  #[test]
  fn a_records_quantities_are_never_training_text() {
    let jsonl = concat!(
      r#"{"label": "recycle", "before": {"name": "Blue Dream 3.5g", "weight_per_item": {"value": 3.5, "unit": "g"}, "#,
      r#""thc_per_item": {"min": 18, "max": 22, "unit": "%"}}, "#,
      r#""after": {"name": "Blue Dream 28g", "weight_per_item": {"value": 28, "unit": "g"}, "#,
      r#""thc_per_item": {"min": 19, "max": 23, "unit": "%"}}}"#
    );
    // Even when a quantity is asked for by name, as a mistaken `--fields` would.
    let found = pair_examples(
      jsonl,
      origin(),
      &fields(&["name", "weight_per_item", "thc_per_item"]),
      Split::Training("calibrate"),
    )
    .unwrap();
    assert_eq!(
      found.examples,
      [PairExample {
        before: "Blue Dream 3.5g".to_owned(),
        after: "Blue Dream 28g".to_owned(),
        same: false
      }]
    );

    let groups = concat!(
      r#"{"catalog": {"name": "Blue Dream", "weight_per_item": {"value": 3.5, "unit": "g"}}, "#,
      r#""listings": [{"name": "Blue Dream Eighth", "weight_per_item": {"value": 0.125, "unit": "oz"}}]}"#
    );
    let found = group_examples(
      groups,
      origin(),
      &fields(&["name", "weight_per_item"]),
      &BTreeSet::new(),
      &[],
    )
    .unwrap()
    .examples;
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].listing, "Blue Dream Eighth");
  }

  #[test]
  fn unsure_labels_and_other_splits_are_skipped_and_counted() {
    let jsonl = [
      r#"{"label": "unsure", "split": "calibrate", "before": {"name": "a"}, "after": {"name": "b"}}"#,
      r#"{"label": "edit", "split": "holdout", "before": {"name": "a"}, "after": {"name": "b"}}"#,
      r#"{"label": "edit", "split": "calibrate", "before": {"name": "a"}, "after": {"name": "b"}}"#,
      r#"{"label": "edit", "before": {"name": "c"}, "after": {"name": "d"}}"#,
    ]
    .join("\n");
    let found = pair_examples(&jsonl, origin(), &fields(&["name"]), Split::Training("calibrate")).unwrap();
    assert_eq!(found.examples.len(), 2, "a pair with no split is training data");
    assert_eq!((found.skipped_label, found.skipped_split), (1, 1));

    let held_out = pair_examples(&jsonl, origin(), &fields(&["name"]), Split::HeldOut("holdout")).unwrap();
    assert_eq!(
      held_out.examples.len(),
      1,
      "only a pair explicitly assigned to the hold-out is held out"
    );
    assert_eq!((held_out.skipped_label, held_out.skipped_split), (1, 2));
  }

  #[test]
  fn a_malformed_line_is_reported_with_its_line_number() {
    let jsonl = "# header\n{\"label\": \"edit\", \"before\": {}, \"after\": {}}\n{\"label\": \"edit\"}\n";
    let error = pair_examples(jsonl, origin(), &fields(&["name"]), Split::Training("calibrate")).unwrap_err();
    assert!(error.to_string().contains("pairs.jsonl, line 3"), "{error}");
  }

  #[test]
  fn each_listing_of_a_catalog_group_pairs_with_the_catalog_entry() {
    let jsonl = concat!(
      r#"{"group": "g1", "catalog": {"name": "Wyld Blueberry Gummies"}, "#,
      r#""listings": [{"name": "WYLD blueberry"}, {"name": "Wyld Blueberry Gummies"}, {"brand": "Wyld"}]}"#
    );
    let examples = group_examples(
      jsonl,
      Path::new("groups.jsonl"),
      &fields(&["name"]),
      &BTreeSet::new(),
      &[],
    )
    .unwrap()
    .examples;
    // The second listing is identical to the catalog entry and the third has no name.
    assert_eq!(
      examples,
      [GroupExample {
        catalog: "Wyld Blueberry Gummies".to_owned(),
        listing: "WYLD blueberry".to_owned()
      }]
    );
  }

  #[test]
  fn the_same_seed_gives_the_same_shuffle_and_another_seed_a_different_one() {
    let shuffled = |seed: u64| {
      let mut items: Vec<usize> = (0..50).collect();
      Rng::new(seed).shuffle(&mut items);
      items
    };
    assert_eq!(shuffled(7), shuffled(7));
    assert_ne!(shuffled(7), shuffled(8));
    let mut sorted = shuffled(7);
    sorted.sort_unstable();
    assert_eq!(sorted, (0..50).collect::<Vec<_>>(), "a shuffle is a permutation");
  }

  #[test]
  fn no_batch_holds_the_same_text_twice_and_every_example_is_used_once() {
    // Three listings of one catalog entry, two of another: the ranking loss must never see two in one batch.
    let texts = [
      ("a", "a1"),
      ("a", "a2"),
      ("a", "a3"),
      ("b", "b1"),
      ("b", "b2"),
      ("c", "a1"),
    ];
    let batches = batches_without_duplicates(&texts, (0..texts.len()).collect(), 4);

    let mut used: Vec<usize> = batches.iter().flatten().copied().collect();
    used.sort_unstable();
    assert_eq!(used, [0, 1, 2, 3, 4, 5]);
    for batch in &batches {
      assert!(batch.len() <= 4);
      let mut seen = HashSet::new();
      for index in batch {
        assert!(
          seen.insert(texts[*index].0) && seen.insert(texts[*index].1),
          "{batches:?}"
        );
      }
    }
    // Example 5 shares a listing text with example 0, so it waits for the second batch, where it fits.
    assert_eq!(batches, [vec![0, 3], vec![1, 4, 5], vec![2]]);
  }

  #[test]
  fn an_epoch_covers_both_kinds_of_example_exactly_once() {
    let pairs: Vec<PairExample> = (0..10)
      .map(|index| PairExample {
        before: format!("before {index}"),
        after: format!("after {index}"),
        same: index % 2 == 0,
      })
      .collect();
    let groups: Vec<GroupExample> = (0..7)
      .map(|index| GroupExample {
        catalog: format!("catalog {index}"),
        listing: format!("listing {index}"),
      })
      .collect();

    let batches = epoch(&pairs, &groups, 4, &mut Rng::new(1));
    let (mut pair_indexes, mut group_indexes) = (Vec::<usize>::new(), Vec::<usize>::new());
    for batch in &batches {
      assert!(!batch.is_empty() && batch.len() <= 4);
      match batch {
        Batch::Pairs(indexes) => pair_indexes.extend(indexes),
        Batch::Groups(indexes) => group_indexes.extend(indexes),
      }
    }
    pair_indexes.sort_unstable();
    group_indexes.sort_unstable();
    assert_eq!(pair_indexes, (0..10).collect::<Vec<_>>());
    assert_eq!(group_indexes, (0..7).collect::<Vec<_>>());
    assert_eq!(batches.len(), 3 + 2);
    assert_eq!(batches, epoch(&pairs, &groups, 4, &mut Rng::new(1)), "deterministic");
  }

  #[test]
  fn pairs_that_share_a_text_still_fill_a_batch_but_catalog_examples_do_not() {
    let pairs: Vec<PairExample> = (0..4)
      .map(|index| PairExample {
        before: "the same listing".to_owned(),
        after: format!("after {index}"),
        same: true,
      })
      .collect();
    assert_eq!(epoch(&pairs, &[], 4, &mut Rng::new(3)).len(), 1);

    // Two listings each of two catalog entries: a batch can hold one listing of each entry, never two of one.
    let groups: Vec<GroupExample> = (0..4)
      .map(|index| GroupExample {
        catalog: format!("catalog entry {}", index % 2),
        listing: format!("listing {index}"),
      })
      .collect();
    let batches = epoch(&[], &groups, 4, &mut Rng::new(3));
    assert_eq!(batches.iter().map(Batch::len).collect::<Vec<_>>(), [2, 2]);
  }

  #[test]
  fn a_catalog_example_with_nothing_to_rank_against_is_left_out_of_the_pass() {
    // Five listings of one entry and one of another: only one batch can hold two examples.
    let groups: Vec<GroupExample> = (0..6)
      .map(|index| GroupExample {
        catalog: if index == 0 {
          "the rare entry"
        } else {
          "the common entry"
        }
        .to_owned(),
        listing: format!("listing {index}"),
      })
      .collect();
    let batches = epoch(&[], &groups, 4, &mut Rng::new(3));
    assert_eq!(batches.iter().map(Batch::len).collect::<Vec<_>>(), [2]);

    let only_one_entry = &groups[1..];
    assert!(epoch(&[], only_one_entry, 4, &mut Rng::new(3)).is_empty());
  }

  #[test]
  fn an_epoch_of_nothing_is_empty() {
    assert!(epoch(&[], &[], 8, &mut Rng::new(0)).is_empty());
  }

  /// A brand held back for measuring must not be learned from through its catalog either.
  #[test]
  fn catalog_groups_of_an_owner_that_is_measured_on_are_left_out_whole() {
    let pairs = concat!(
      r#"{"label": "edit", "split": "calibrate", "owner": "brand-1", "before": {"name": "a"}, "after": {"name": "b"}}"#,
      "\n",
      r#"{"label": "edit", "split": "validation", "owner": "brand-2", "before": {"name": "a"}, "after": {"name": "b"}}"#,
      "\n",
      r#"{"label": "recycle", "split": "holdout", "owner": "brand-3", "before": {"name": "a"}, "after": {"name": "b"}}"#,
      "\n",
      r#"{"label": "edit", "split": "holdout", "before": {"name": "a"}, "after": {"name": "b"}}"#,
    );
    let measured_on = owners_in(pairs, Path::new("pairs.jsonl"), &["validation", "holdout"]).unwrap();
    assert_eq!(
      measured_on.iter().map(String::as_str).collect::<Vec<_>>(),
      ["brand-2", "brand-3"]
    );

    let group = |owner: Option<&str>, name: &str| {
      let owner = owner.map_or_else(String::new, |owner| format!(r#""owner": "{owner}", "#));
      format!(r#"{{{owner}"catalog": {{"name": "{name}"}}, "listings": [{{"name": "{name} 100mg"}}]}}"#)
    };
    let groups = [
      group(Some("brand-1"), "gummies"),
      group(Some("brand-2"), "flower"),
      group(Some("brand-3"), "vape"),
      group(None, "tincture"),
    ]
    .join("\n");
    let found = group_examples(
      &groups,
      Path::new("groups.jsonl"),
      &fields(&["name"]),
      &measured_on,
      &[],
    )
    .unwrap();
    let kept: Vec<&str> = found.examples.iter().map(|example| example.catalog.as_str()).collect();
    assert_eq!(kept, ["gummies", "tincture"]);
    assert_eq!((found.excluded_groups, found.groups_without_owner), (2, 1));
  }

  /// A group can say which split it is in itself. The held-out ones are what a match threshold is measured on
  /// (`calibrate-match --split holdout`), so they are not trained on, whoever owns them.
  #[test]
  fn a_catalog_group_in_a_held_out_split_is_not_trained_on_and_one_without_a_split_is() {
    let group = |split: Option<&str>, name: &str| {
      let split = split.map_or_else(String::new, |split| format!(r#""split": "{split}", "#));
      format!(
        r#"{{{split}"owner": "brand-9", "catalog": {{"name": "{name}"}}, "listings": [{{"name": "{name} 10pk"}}]}}"#
      )
    };
    let groups = [
      group(Some("calibrate"), "gummies"),
      group(Some("holdout"), "flower"),
      group(Some("validation"), "vape"),
      group(None, "tincture"),
    ]
    .join("\n");
    let found = group_examples(
      &groups,
      Path::new("groups.jsonl"),
      &fields(&["name"]),
      &BTreeSet::new(),
      &["holdout", "validation"],
    )
    .unwrap();
    let kept: Vec<&str> = found.examples.iter().map(|example| example.catalog.as_str()).collect();
    assert_eq!(kept, ["gummies", "tincture"]);
    assert_eq!((found.excluded_groups, found.groups_without_owner), (2, 0));

    let all = group_examples(
      &groups,
      Path::new("groups.jsonl"),
      &fields(&["name"]),
      &BTreeSet::new(),
      &[],
    )
    .unwrap();
    assert_eq!((all.examples.len(), all.excluded_groups), (4, 0));
  }
}
