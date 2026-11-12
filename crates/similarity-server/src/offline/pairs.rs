//! Labeled before/after pairs, as the two file formats hold them.
//!
//! - The **strict** format is four keys per line (`label`, `note`, `before`, `after`) and refuses anything else. It
//!   is what a person writes by hand, and a typo in it should be an error.
//! - The **full** format is what `docs/data-request-prompt.md` asks a data export to produce (`pairs.full.jsonl`):
//!   the same pair with its bookkeeping (`pair_id`, `owner`, `split`, `variant_change`, how it was labeled) and
//!   with every field the source had, most of which the schema does not define. It is read leniently, and what is
//!   left out is counted rather than passed over in silence.
//!
//! Both produce the same [`Pair`], so everything downstream is one code path.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use similarity_core::record::hex_prefix;
use similarity_core::{Quantity, Schema};

use super::RecordFields;

/// What a labeled pair is an example of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Label {
  /// The same record, legitimately edited. Should be accepted.
  Edit,
  /// A different record under the old id. Should be rejected.
  Recycle,
}

impl Label {
  /// The label as the files spell it.
  pub fn as_str(self) -> &'static str {
    match self {
      Self::Edit => "edit",
      Self::Recycle => "recycle",
    }
  }
}

/// How strong the evidence behind a label is, as the data request grades it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
  /// Weak or indirect evidence.
  Low,
  /// Reasonable evidence.
  Medium,
  /// A moderator action, an identifier change, or the like.
  High,
}

impl Confidence {
  /// The grade as the files spell it.
  pub fn as_str(self) -> &'static str {
    match self {
      Self::Low => "low",
      Self::Medium => "medium",
      Self::High => "high",
    }
  }
}

impl FromStr for Confidence {
  type Err = String;

  fn from_str(text: &str) -> Result<Self, Self::Err> {
    match text {
      "low" => Ok(Self::Low),
      "medium" => Ok(Self::Medium),
      "high" => Ok(Self::High),
      other => Err(format!("`{other}` is not a confidence; use high, medium, or low")),
    }
  }
}

/// One before/after pair, ready to score.
#[derive(Debug, Clone, PartialEq)]
pub struct Pair {
  /// What identifies the pair across runs: its `pair_id` if the file gave one, otherwise a fingerprint of its
  /// content, so that the same pair scored twice can still be matched up.
  pub id: String,
  /// What the pair is an example of. `None` for any label other than `edit` and `recycle` (`unsure`, typically):
  /// such a pair is scored and shown, and never counted as right or wrong.
  pub label: Option<Label>,
  /// The label exactly as the file wrote it.
  pub label_text: String,
  /// A short description, shown in the report.
  pub note: String,
  /// Which part of the data the pair belongs to (`calibrate`, `holdout`).
  pub split: Option<String>,
  /// Who owns the record, as a pseudonym. Pairs from one owner are not independent of each other.
  pub owner: Option<String>,
  /// The kind of same-brand variant change this is (`flavor`, `size`, ...), if it is one.
  pub variant_change: Option<String>,
  /// The evidence the label rests on.
  pub label_source: Option<String>,
  /// How strong that evidence is, as written.
  pub label_confidence: Option<String>,
  /// The record before the change.
  pub before: RecordFields,
  /// The record after the change.
  pub after: RecordFields,
  /// The record as it was first created, if the pair says. It makes the pair one step of a history: created as
  /// `anchor`, now stored as `before`, with `after` arriving. A field `before` lacks is then scored against it,
  /// and drift is measured from it, as the server does for a stored record (D8).
  pub anchor: Option<RecordFields>,
}

impl Pair {
  /// The top-level category of the record before the change: the category text up to the first ` > `, which is
  /// the separator callers are told to build the path with.
  pub fn category(&self) -> Option<&str> {
    let path = self.before.texts.get("category")?;
    let top = path.split(" > ").next().unwrap_or(path).trim();
    (!top.is_empty()).then_some(top)
  }
}

/// A fingerprint of a pair's content: the first 16 hex digits of the SHA-256 of its records as JSON.
///
/// Keys are written in sorted order, so the same content gives the same id however the file listed it. The anchor
/// is hashed only when there is one, so a pair without one has the id it had before pairs could carry an anchor.
fn content_id(before: &RecordFields, after: &RecordFields, anchor: Option<&RecordFields>) -> anyhow::Result<String> {
  let bytes = match anchor {
    Some(anchor) => serde_json::to_vec(&(before, after, anchor)),
    None => serde_json::to_vec(&(before, after)),
  }
  .context("a pair does not serialize")?;
  Ok(hex_prefix(&Sha256::digest(&bytes), 8))
}

/// Why a key of a record was left out of the comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ignored {
  /// The schema defines neither a text field nor a quantity of that name.
  NotInSchema,
  /// The schema defines the name, but as the other kind: text where a quantity belongs, or the reverse.
  WrongKind,
}

impl fmt::Display for Ignored {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::NotInSchema => "not in the schema",
      Self::WrongKind => "the wrong kind of value",
    })
  }
}

/// What a set of pairs was read from, and what was left out along the way.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Input {
  /// The SHA-256 of the file's text. Two runs are over the same pairs only if this and the selection agree.
  pub sha256: String,
  /// The split the pairs were restricted to, if any.
  pub split: Option<String>,
  /// The least confidence a labeled pair needed, if any.
  pub min_confidence: Option<Confidence>,
  /// Record keys that were not compared, with how many values each lost.
  pub ignored: BTreeMap<(String, Ignored), usize>,
  /// Pairs left out because they belong to another split.
  pub skipped_split: usize,
  /// Labeled pairs left out because their label's confidence is below the minimum.
  pub skipped_confidence: usize,
}

/// The pairs of one file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Pairs {
  /// The pairs, in file order.
  pub pairs: Vec<Pair>,
  /// Where they came from.
  pub input: Input,
}

/// The lines of a JSONL document that hold data, with their one-based line numbers.
pub(super) fn data_lines(jsonl: &str) -> impl Iterator<Item = (usize, &str)> {
  jsonl
    .lines()
    .enumerate()
    .filter(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
    .map(|(index, line)| (index.saturating_add(1), line))
}

/// Refuses a file in which two pairs share an id: they could not be told apart when two runs are compared, and a
/// pair listed twice counts twice.
fn unique(pairs: &[(usize, Pair)]) -> anyhow::Result<()> {
  let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
  for (line, pair) in pairs {
    if let Some(first) = seen.insert(pair.id.as_str(), *line) {
      bail!(
        "lines {first} and {line} are the same pair (`{}`); remove one, or give them different `pair_id`s",
        pair.id
      );
    }
  }
  Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictLine {
  label: Label,
  #[serde(default)]
  note: String,
  before: RecordFields,
  after: RecordFields,
  #[serde(default)]
  anchor: Option<RecordFields>,
}

/// Parses a calibration file in the strict format: one JSON object per line with `label`, `note`, `before`, and
/// `after`, optionally `anchor` (the record as first created), and nothing else; blank lines and `#` comments
/// ignored.
///
/// # Errors
///
/// Returns an error naming the line if a line is not a valid pair, or naming both lines if a pair appears twice.
pub fn parse_pairs(jsonl: &str) -> anyhow::Result<Pairs> {
  let mut pairs = Vec::new();
  for (number, line) in data_lines(jsonl) {
    let parsed: StrictLine =
      serde_json::from_str(line).with_context(|| format!("line {number} is not a valid pair"))?;
    let pair = Pair {
      id: content_id(&parsed.before, &parsed.after, parsed.anchor.as_ref())?,
      label: Some(parsed.label),
      label_text: parsed.label.as_str().to_owned(),
      note: parsed.note,
      split: None,
      owner: None,
      variant_change: None,
      label_source: None,
      label_confidence: None,
      before: parsed.before,
      after: parsed.after,
      anchor: parsed.anchor,
    };
    pairs.push((number, pair));
  }
  unique(&pairs)?;
  Ok(Pairs {
    pairs: pairs.into_iter().map(|(_, pair)| pair).collect(),
    input: Input {
      sha256: hex_prefix(&Sha256::digest(jsonl.as_bytes()), 32),
      ..Input::default()
    },
  })
}

// Unknown keys are ignored on purpose: the full format carries bookkeeping that only matters to whoever
// assembled it, and more of it will be added over time.
#[derive(Deserialize)]
struct FullLine {
  #[serde(default)]
  pair_id: Option<String>,
  label: String,
  #[serde(default)]
  note: Option<String>,
  #[serde(default)]
  split: Option<String>,
  #[serde(default)]
  owner: Option<String>,
  #[serde(default)]
  variant_change: Option<String>,
  #[serde(default)]
  label_source: Option<String>,
  #[serde(default)]
  label_confidence: Option<String>,
  before: BTreeMap<String, Value>,
  after: BTreeMap<String, Value>,
  #[serde(default)]
  anchor: Option<BTreeMap<String, Value>>,
}

/// Which pairs of a full file to score.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Selection<'a> {
  /// Only pairs of this split. `None` takes every pair, whatever its split.
  pub split: Option<&'a str>,
  /// Only labeled pairs whose label is at least this well evidenced. A pair with no confidence recorded is kept:
  /// there is nothing to filter it on. Pairs with any other label are never filtered by this.
  pub min_confidence: Option<Confidence>,
}

/// Keeps the keys of `record` that the schema can compare, and counts the rest.
///
/// A `null` is "no value", as an absent key is, and is not counted as anything.
pub(super) fn project(
  record: BTreeMap<String, Value>,
  schema: &Schema,
  ignored: &mut BTreeMap<(String, Ignored), usize>,
) -> anyhow::Result<RecordFields> {
  let mut fields = RecordFields::default();
  for (key, value) in record {
    let (is_field, is_quantity) = (schema.field(&key).is_some(), schema.quantity(&key).is_some());
    match value {
      Value::Null => {}
      Value::String(text) if is_field => {
        fields.texts.insert(key, text);
      }
      Value::Object(_) if is_quantity => {
        let quantity: Quantity = serde_json::from_value(value).with_context(|| format!("quantity `{key}`"))?;
        fields.quantities.insert(key, quantity);
      }
      _ => {
        let reason = if is_field || is_quantity {
          Ignored::WrongKind
        } else {
          Ignored::NotInSchema
        };
        let count = ignored.entry((key, reason)).or_default();
        *count = count.saturating_add(1);
      }
    }
  }
  Ok(fields)
}

/// Parses a calibration file in the full format (`pairs.full.jsonl`), keeping what `schema` can compare.
///
/// Each record is projected onto the schema: a key the schema defines as a text field is kept when its value is a
/// string, a key it defines as a quantity is kept when its value is an object, and everything else is left out
/// and counted in [`Input::ignored`], so that a misspelled key shows up as a count and not as a weaker score.
///
/// # Errors
///
/// Returns an error naming the line if a line is not a JSON object with `label`, `before`, and `after`, if a
/// quantity is malformed, or if `label_confidence` is not one of `high`, `medium`, and `low`; and naming both
/// lines if two pairs share an id.
pub fn parse_full_pairs(jsonl: &str, schema: &Schema, selection: Selection<'_>) -> anyhow::Result<Pairs> {
  let mut input = Input {
    sha256: hex_prefix(&Sha256::digest(jsonl.as_bytes()), 32),
    split: selection.split.map(str::to_owned),
    min_confidence: selection.min_confidence,
    ..Input::default()
  };
  let mut pairs = Vec::new();
  for (number, line) in data_lines(jsonl) {
    let invalid = || format!("line {number} is not a valid pair");
    let parsed: FullLine = serde_json::from_str(line).with_context(invalid)?;
    if selection.split.is_some() && selection.split != parsed.split.as_deref() {
      input.skipped_split = input.skipped_split.saturating_add(1);
      continue;
    }
    let label = match parsed.label.as_str() {
      "edit" => Some(Label::Edit),
      "recycle" => Some(Label::Recycle),
      _ => None,
    };
    let confidence = parsed
      .label_confidence
      .as_deref()
      .map(Confidence::from_str)
      .transpose()
      .map_err(anyhow::Error::msg)
      .with_context(invalid)?;
    if let (Some(_), Some(found), Some(wanted)) = (label, confidence, selection.min_confidence)
      && found < wanted
    {
      input.skipped_confidence = input.skipped_confidence.saturating_add(1);
      continue;
    }

    let before = project(parsed.before, schema, &mut input.ignored).with_context(invalid)?;
    let after = project(parsed.after, schema, &mut input.ignored).with_context(invalid)?;
    let anchor = parsed
      .anchor
      .map(|anchor| project(anchor, schema, &mut input.ignored))
      .transpose()
      .with_context(invalid)?;
    let id = match parsed.pair_id.filter(|id| !id.trim().is_empty()) {
      Some(id) => id,
      None => content_id(&before, &after, anchor.as_ref())?,
    };
    let pair = Pair {
      id,
      label,
      label_text: parsed.label,
      note: parsed.note.unwrap_or_default(),
      split: parsed.split,
      owner: parsed.owner,
      variant_change: parsed.variant_change,
      label_source: parsed.label_source,
      label_confidence: parsed.label_confidence,
      before,
      after,
      anchor,
    };
    pairs.push((number, pair));
  }
  unique(&pairs)?;
  Ok(Pairs {
    pairs: pairs.into_iter().map(|(_, pair)| pair).collect(),
    input,
  })
}

#[cfg(test)]
mod tests {
  use similarity_core::{FieldSpec, QuantitySpec};

  use super::*;

  fn schema() -> Schema {
    let field = |name: &str| FieldSpec {
      name: name.to_owned(),
      weight: 1.0,
      required: false,
      reject_below: None,
    };
    Schema::new(vec![field("name"), field("category"), field("description")])
      .unwrap()
      .with_quantities(vec![QuantitySpec {
        name: "weight_per_item".to_owned(),
        reject_above_factor: None,
        reject_incomparable: false,
      }])
      .unwrap()
  }

  fn everything() -> Selection<'static> {
    Selection::default()
  }

  #[test]
  fn pairs_parse_from_jsonl_with_comments_and_blank_lines() {
    let text = r#"
      # ordinary edits
      {"label": "edit", "note": "potency added", "before": {"name": "Gummies"}, "after": {"name": "Gummies 100mg"}}

      {"label": "recycle", "before": {"name": "Gummies"}, "after": {"name": "Vape"}}
    "#;
    let parsed = parse_pairs(text).unwrap();
    let pairs = &parsed.pairs;
    assert_eq!(pairs.len(), 2);
    assert_eq!(
      (pairs[0].label, pairs[0].note.as_str()),
      (Some(Label::Edit), "potency added")
    );
    assert_eq!(pairs[1].after.texts["name"], "Vape");
    assert!(pairs[1].after.quantities.is_empty());
    assert_eq!(parsed.input.sha256.len(), 64);
    assert!(parsed.input.ignored.is_empty());
  }

  #[test]
  fn a_bad_line_is_reported_by_number() {
    let text = "{\"label\": \"edit\", \"before\": {}, \"after\": {}}\n{\"label\": \"sideways\"}";
    let error = parse_pairs(text).unwrap_err();
    assert!(format!("{error:#}").contains("line 2"), "{error:#}");
    assert!(parse_pairs(r#"{"label": "edit", "before": {}, "after": {}, "extra": 1}"#).is_err());
  }

  #[test]
  fn a_pair_without_an_id_is_identified_by_its_content() {
    let line = |note: &str, before: &str| {
      format!(r#"{{"label": "edit", "note": "{note}", "before": {before}, "after": {{"name": "b"}}}}"#)
    };
    let id = |text: &str| parse_pairs(text).unwrap().pairs[0].id.clone();

    let plain = id(&line("one", r#"{"name": "a", "category": "c"}"#));
    assert_eq!(plain.len(), 16);
    assert_eq!(
      plain,
      id(&line("another note", r#"{"category": "c", "name": "a"}"#)),
      "neither the note nor the order of keys is content"
    );
    assert_ne!(plain, id(&line("one", r#"{"name": "a", "category": "d"}"#)));
    assert_ne!(
      plain,
      id(&line(
        "one",
        r#"{"name": "a", "category": "c", "weight_per_item": {"value": 1, "unit": "g"}}"#
      ))
    );
  }

  #[test]
  fn the_same_pair_twice_is_refused_with_both_line_numbers() {
    let pair = r#"{"label": "edit", "before": {"name": "a"}, "after": {"name": "b"}}"#;
    let error = parse_pairs(&format!("{pair}\n# again\n{pair}"))
      .unwrap_err()
      .to_string();
    assert!(error.contains("lines 1 and 3"), "{error}");

    let full = |id: &str| {
      format!(r#"{{"pair_id": "{id}", "label": "edit", "before": {{"name": "a"}}, "after": {{"name": "b"}}}}"#)
    };
    let error = parse_full_pairs(&format!("{}\n{}", full("p1"), full("p1")), &schema(), everything()).unwrap_err();
    assert!(error.to_string().contains("`p1`"), "{error}");
    // Identical content under two ids is two pairs: the ids say so.
    assert_eq!(
      parse_full_pairs(&format!("{}\n{}", full("p1"), full("p2")), &schema(), everything())
        .unwrap()
        .pairs
        .len(),
      2
    );
  }

  const FULL: &str = concat!(
    r#"{"pair_id": "p1", "product": "x", "owner": "brand-1", "edited_at": "2026-03-14T17:02:11Z", "#,
    r#""label": "recycle", "label_source": "moderator_revert", "label_confidence": "high", "#,
    r#""variant_change": null, "split": "calibrate", "note": "gummies became a vape", "#,
    r#""before": {"name": "Gummies", "category": "Edibles > Gummies", "strain": "n/a", "strain_id": 7, "#,
    r#""weight_per_item": {"value": 40, "unit": "g"}, "weight_per_item_one_sided": true, "description": null}, "#,
    r#""after": {"name": "Vape", "category": "Vape Pens", "strain": "Sour Diesel", "#,
    r#""weight_per_item": "1 g", "descripton": "A cartridge."}}"#,
  );

  #[test]
  fn a_full_line_keeps_its_bookkeeping_and_only_what_the_schema_can_compare() {
    let parsed = parse_full_pairs(FULL, &schema(), everything()).unwrap();
    let [pair] = parsed.pairs.as_slice() else {
      panic!("one pair expected")
    };
    assert_eq!(pair.id, "p1");
    assert_eq!(
      (pair.label, pair.label_text.as_str()),
      (Some(Label::Recycle), "recycle")
    );
    assert_eq!(pair.owner.as_deref(), Some("brand-1"));
    assert_eq!(pair.split.as_deref(), Some("calibrate"));
    assert_eq!(pair.variant_change, None, "a null is no value");
    assert_eq!(pair.label_source.as_deref(), Some("moderator_revert"));
    assert_eq!(pair.category(), Some("Edibles"));

    assert_eq!(
      pair.before.texts.keys().collect::<Vec<_>>(),
      ["category", "name"],
      "a null description is no description"
    );
    assert_eq!(pair.before.quantities["weight_per_item"], Quantity::single(40.0, "g"));
    assert_eq!(pair.after.texts.keys().collect::<Vec<_>>(), ["category", "name"]);
    assert!(
      pair.after.quantities.is_empty(),
      "a quantity rendered as text is not one"
    );
  }

  #[test]
  fn what_a_full_line_loses_is_counted_by_key_and_reason() {
    let parsed = parse_full_pairs(FULL, &schema(), everything()).unwrap();
    let ignored: Vec<(&str, Ignored, usize)> = parsed
      .input
      .ignored
      .iter()
      .map(|((key, reason), count)| (key.as_str(), *reason, *count))
      .collect();
    assert_eq!(
      ignored,
      [
        // The misspelling is the one that matters: without the count it would only be a weaker score.
        ("descripton", Ignored::NotInSchema, 1),
        ("strain", Ignored::NotInSchema, 2),
        ("strain_id", Ignored::NotInSchema, 1),
        ("weight_per_item", Ignored::WrongKind, 1),
        ("weight_per_item_one_sided", Ignored::NotInSchema, 1),
      ]
    );
  }

  #[test]
  fn a_split_and_a_minimum_confidence_select_pairs_and_count_the_rest() {
    let line = |id: &str, label: &str, split: &str, confidence: &str| {
      format!(
        r#"{{"pair_id": "{id}", "label": "{label}", "split": "{split}", "label_confidence": "{confidence}", "before": {{"name": "a {id}"}}, "after": {{"name": "b"}}}}"#
      )
    };
    let text = [
      line("p1", "edit", "calibrate", "high"),
      line("p2", "edit", "calibrate", "low"),
      line("p3", "recycle", "holdout", "high"),
      line("p4", "unsure", "calibrate", "low"),
      r#"{"pair_id": "p5", "label": "edit", "before": {"name": "a"}, "after": {"name": "b"}}"#.to_owned(),
    ]
    .join("\n");
    let ids = |selection: Selection<'_>| {
      let parsed = parse_full_pairs(&text, &schema(), selection).unwrap();
      let ids: Vec<String> = parsed.pairs.iter().map(|pair| pair.id.clone()).collect();
      (
        ids.join(" "),
        parsed.input.skipped_split,
        parsed.input.skipped_confidence,
      )
    };

    assert_eq!(ids(everything()), ("p1 p2 p3 p4 p5".to_owned(), 0, 0));
    let calibrate = Selection {
      split: Some("calibrate"),
      min_confidence: None,
    };
    assert_eq!(
      ids(calibrate),
      ("p1 p2 p4".to_owned(), 2, 0),
      "a pair with no split is in no split"
    );
    let trusted = Selection {
      split: None,
      min_confidence: Some(Confidence::Medium),
    };
    // The unsure pair is not filtered on confidence (it is never counted), nor is the pair with none recorded.
    assert_eq!(ids(trusted), ("p1 p3 p4 p5".to_owned(), 0, 1));

    let unknown = line("p9", "edit", "calibrate", "certain");
    let error = parse_full_pairs(&unknown, &schema(), everything()).unwrap_err();
    assert!(
      format!("{error:#}").contains("`certain` is not a confidence"),
      "{error:#}"
    );
  }

  #[test]
  fn any_other_label_is_kept_unlabeled() {
    let text = r#"{"pair_id": "p1", "label": "unsure", "before": {"name": "a"}, "after": {"name": "b"}}"#;
    let parsed = parse_full_pairs(text, &schema(), everything()).unwrap();
    assert_eq!(
      (parsed.pairs[0].label, parsed.pairs[0].label_text.as_str()),
      (None, "unsure")
    );
  }

  #[test]
  fn a_malformed_quantity_in_a_full_line_is_an_error_not_an_ignored_key() {
    let text = r#"{"label": "edit", "before": {"name": "a", "weight_per_item": {"value": 1}}, "after": {"name": "b"}}"#;
    let error = parse_full_pairs(text, &schema(), everything()).unwrap_err();
    let shown = format!("{error:#}");
    assert!(shown.contains("line 1") && shown.contains("weight_per_item"), "{shown}");
  }

  #[test]
  fn a_pair_may_carry_the_record_as_it_was_first_created() {
    let line = concat!(
      r#"{"label": "recycle", "note": "blanked, then refilled", "anchor": {"name": "Gummies", "description": "ten"}, "#,
      r#""before": {"name": "Gummies"}, "after": {"name": "Gummies", "description": "one gram"}}"#,
    );
    let strict = parse_pairs(line).unwrap();
    let anchor = strict.pairs[0].anchor.as_ref().unwrap();
    assert_eq!(anchor.texts["description"], "ten");

    let full = parse_full_pairs(line, &schema(), everything()).unwrap();
    assert_eq!(full.pairs[0].anchor, strict.pairs[0].anchor);
    assert_eq!(
      full.pairs[0].id, strict.pairs[0].id,
      "both formats hash the same content"
    );

    // The anchor is part of what the pair is: the same step from another first version is another pair.
    let other = parse_pairs(&line.replace(r#""description": "ten""#, r#""description": "twenty""#)).unwrap();
    assert_ne!(other.pairs[0].id, strict.pairs[0].id);
    let none = parse_pairs(r#"{"label": "edit", "before": {"name": "a"}, "after": {"name": "b"}}"#).unwrap();
    assert_eq!(none.pairs[0].anchor, None);
  }

  /// Runs are compared by pair id. An id that changed because anchors exist now would make every pair of every
  /// earlier run look new.
  #[test]
  fn a_pair_without_an_anchor_keeps_the_id_it_had() {
    let sample = include_str!("../../../../data/calibration.sample.jsonl");
    let parsed = parse_pairs(sample).unwrap();
    assert_eq!(parsed.pairs[0].note, "potency added to the name");
    assert_eq!(
      parsed.pairs[0].id, "408d16dda7b0501e",
      "the id this pair had before pairs could carry an anchor"
    );
  }

  #[test]
  fn an_anchor_is_reduced_to_the_schema_and_its_dropped_keys_counted() {
    let line = concat!(
      r#"{"label": "edit", "anchor": {"name": "a", "sku": "x1", "weight_per_item": "heavy"}, "#,
      r#""before": {"name": "a"}, "after": {"name": "b"}}"#,
    );
    let parsed = parse_full_pairs(line, &schema(), everything()).unwrap();
    let anchor = parsed.pairs[0].anchor.as_ref().unwrap();
    assert_eq!(anchor.texts.len(), 1);
    assert!(anchor.quantities.is_empty());
    assert_eq!(parsed.input.ignored[&("sku".to_owned(), Ignored::NotInSchema)], 1);
    assert_eq!(
      parsed.input.ignored[&("weight_per_item".to_owned(), Ignored::WrongKind)],
      1
    );

    let malformed = line.replace(r#""heavy""#, r#"{"value": 1}"#);
    let error = format!(
      "{:#}",
      parse_full_pairs(&malformed, &schema(), everything()).unwrap_err()
    );
    assert!(error.contains("line 1") && error.contains("weight_per_item"), "{error}");
  }

  #[test]
  fn a_strict_pair_with_any_fifth_key_but_anchor_is_refused() {
    let with = |key: &str| {
      format!(r#"{{"label": "edit", "{key}": {{"name": "a"}}, "before": {{"name": "a"}}, "after": {{"name": "b"}}}}"#)
    };
    assert!(parse_pairs(&with("anchor")).is_ok());
    let error = format!("{:#}", parse_pairs(&with("anchr")).unwrap_err());
    assert!(error.contains("unknown field `anchr`"), "{error}");
  }

  #[test]
  fn the_top_level_category_is_the_text_before_the_first_separator() {
    let pair = |category: Option<&str>| {
      let mut before = RecordFields::default();
      if let Some(category) = category {
        before.texts.insert("category".to_owned(), category.to_owned());
      }
      Pair {
        id: "p".to_owned(),
        label: None,
        label_text: String::new(),
        note: String::new(),
        split: None,
        owner: None,
        variant_change: None,
        label_source: None,
        label_confidence: None,
        before,
        after: RecordFields::default(),
        anchor: None,
      }
    };
    assert_eq!(pair(Some("Edibles > Gummies > Vegan")).category(), Some("Edibles"));
    assert_eq!(pair(Some("Tinctures")).category(), Some("Tinctures"));
    assert_eq!(pair(Some("  ")).category(), None);
    assert_eq!(pair(None).category(), None);
  }
}
