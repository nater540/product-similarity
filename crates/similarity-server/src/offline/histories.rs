//! Whole edit histories, for calibrating drift: `sequences.jsonl`.
//!
//! A calibration pair is one edit. The drift check is about many: a record moved a long way from what it first
//! was, in steps that each look fine. Measuring that needs every version of a product, in order.
//!
//! One JSON object per line:
//!
//! ```json
//! {"product": "h7", "brand_id": "b12", "label": "recycle", "recycled_at": 3,
//!  "versions": [{"at": "2025-11-02T09:00:00Z", "fields": {"name": "Blueberry Gummies"}}, ...]}
//! ```
//!
//! Each version's `fields` is a record, reduced to what the schema compares as a pair of the full format is, with
//! everything left out counted. The keys of a line itself are fixed, so that a misspelled `recycled_at` is an
//! error and not a history that silently says nothing about where it went wrong.

use std::collections::BTreeMap;
use std::str::FromStr;

use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use similarity_core::Schema;
use similarity_core::record::hex_prefix;

use super::RecordFields;
use super::pairs::{Confidence, Input, Label, data_lines, project};

/// One version of a product.
#[derive(Debug, Clone, PartialEq)]
pub struct Version {
  /// When it was saved, as the file wrote it. Kept as text and never parsed: the order of the file is the order.
  pub at: Option<String>,
  /// The record at that version.
  pub fields: RecordFields,
}

/// One product's history.
#[derive(Debug, Clone, PartialEq)]
pub struct History {
  /// What identifies the history: the product, as a pseudonym.
  pub id: String,
  /// Who owns the product, as a pseudonym. Histories of one brand are not independent of each other.
  pub brand_id: Option<String>,
  /// `Edit` for a product that stayed itself throughout, `Recycle` for one that became another. `None` for any
  /// other label: such a history is replayed and shown, and never counted.
  pub label: Option<Label>,
  /// The label as the file wrote it.
  pub label_text: String,
  /// The evidence the label rests on.
  pub label_source: Option<String>,
  /// How strong that evidence is.
  pub label_confidence: Option<Confidence>,
  /// For a recycled product, the index in `versions` of the first version that is no longer the product. What
  /// comes before it is a legitimate history; a history without it is known to be recycled somewhere.
  pub recycled_at: Option<usize>,
  /// The first version here is not the product as created, and how it was created is not known. Its drift is
  /// understated by however far it had already come.
  pub truncated: bool,
  /// The product as created, when the first version here is not that.
  pub original: Option<RecordFields>,
  /// Every version, in order. At least two.
  pub versions: Vec<Version>,
}

/// The histories of a file, and what was left out of them.
#[derive(Debug, Clone, PartialEq)]
pub struct Histories {
  /// The histories to replay.
  pub histories: Vec<History>,
  /// The file's fingerprint, the selection, and the record keys that were not compared.
  pub input: Input,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VersionLine {
  #[serde(default)]
  at: Option<String>,
  fields: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Line {
  product: String,
  #[serde(default)]
  brand_id: Option<String>,
  label: String,
  #[serde(default)]
  recycled_at: Option<usize>,
  #[serde(default)]
  label_source: Option<String>,
  #[serde(default)]
  label_confidence: Option<String>,
  #[serde(default)]
  truncated: bool,
  #[serde(default)]
  original: Option<BTreeMap<String, Value>>,
  versions: Vec<VersionLine>,
}

/// Parses a file of histories, keeping of each version what `schema` can compare.
///
/// A labeled history whose `label_confidence` is below `min_confidence` is left out and counted, as a pair is.
///
/// # Errors
///
/// Returns an error naming the line if a line is not a history (an unknown key included), if it has fewer than
/// two versions, if `recycled_at` does not point at a version after the first or is given for a history that is
/// not a recycle, or if a quantity is malformed; and naming both lines if two histories share a product.
pub fn parse_histories(jsonl: &str, schema: &Schema, min_confidence: Option<Confidence>) -> anyhow::Result<Histories> {
  let mut input = Input {
    sha256: hex_prefix(&Sha256::digest(jsonl.as_bytes()), 32),
    min_confidence,
    ..Input::default()
  };
  let mut histories = Vec::new();
  let mut seen: BTreeMap<String, usize> = BTreeMap::new();
  for (number, line) in data_lines(jsonl) {
    let invalid = || format!("line {number} is not a valid history");
    let parsed: Line = serde_json::from_str(line).with_context(invalid)?;
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
    if let Some(first) = seen.insert(parsed.product.clone(), number) {
      bail!(
        "lines {first} and {number} are the same product (`{}`); a product has one history",
        parsed.product
      );
    }
    if parsed.versions.len() < 2 {
      bail!(
        "line {number}: `{}` has {} version(s); a history is at least two",
        parsed.product,
        parsed.versions.len()
      );
    }
    match parsed.recycled_at {
      Some(_) if label != Some(Label::Recycle) => bail!(
        "line {number}: `{}` has `recycled_at` and is labeled `{}`; only a recycle has a version it was recycled at",
        parsed.product,
        parsed.label
      ),
      Some(at) if at == 0 || at >= parsed.versions.len() => bail!(
        "line {number}: `{}` has `recycled_at` {at}; it is the index of a version after the first, from 1 to {}",
        parsed.product,
        parsed.versions.len().saturating_sub(1)
      ),
      _ => {}
    }
    if let (Some(_), Some(found), Some(wanted)) = (label, confidence, min_confidence)
      && found < wanted
    {
      input.skipped_confidence = input.skipped_confidence.saturating_add(1);
      continue;
    }

    let original = parsed
      .original
      .map(|original| project(original, schema, &mut input.ignored))
      .transpose()
      .with_context(|| format!("line {number}: the original of `{}`", parsed.product))?;
    let mut versions = Vec::with_capacity(parsed.versions.len());
    for (index, version) in parsed.versions.into_iter().enumerate() {
      let fields = project(version.fields, schema, &mut input.ignored)
        .with_context(|| format!("line {number}: version {index} of `{}`", parsed.product))?;
      versions.push(Version { at: version.at, fields });
    }
    histories.push(History {
      id: parsed.product,
      brand_id: parsed.brand_id,
      label,
      label_text: parsed.label,
      label_source: parsed.label_source,
      label_confidence: confidence,
      recycled_at: parsed.recycled_at,
      truncated: parsed.truncated,
      original,
      versions,
    });
  }
  Ok(Histories { histories, input })
}

#[cfg(test)]
mod tests {
  use similarity_core::{FieldSpec, QuantitySpec};

  use super::*;
  use crate::offline::pairs::Ignored;

  fn schema() -> Schema {
    let field = |name: &str| FieldSpec {
      name: name.to_owned(),
      weight: 1.0,
      required: false,
      reject_below: None,
    };
    Schema::new(vec![field("name"), field("description")])
      .unwrap()
      .with_quantities(vec![QuantitySpec {
        name: "weight".to_owned(),
        reject_above_factor: None,
        reject_incomparable: false,
      }])
      .unwrap()
  }

  fn version(name: &str) -> String {
    format!(r#"{{"at": "2026-01-01T00:00:00Z", "fields": {{"name": "{name}"}}}}"#)
  }

  fn history(product: &str, rest: &str, versions: &[&str]) -> String {
    let versions: Vec<String> = versions.iter().map(|name| version(name)).collect();
    format!(
      r#"{{"product": "{product}", {rest}"versions": [{}]}}"#,
      versions.join(", ")
    )
  }

  #[test]
  fn a_history_is_every_version_in_order_reduced_to_what_the_schema_compares() {
    let text = [
      "# synthetic".to_owned(),
      String::new(),
      concat!(
        r#"{"product": "h1", "brand_id": "b1", "label": "recycle", "recycled_at": 2, "label_source": "moderator", "#,
        r#""label_confidence": "high", "original": {"name": "gummies", "sku": "x"}, "#,
        r#""versions": [{"fields": {"name": "gummies 100mg", "weight": {"value": 40, "unit": "g"}, "strain": "s"}}, "#,
        r#"{"at": "2026-02-01T00:00:00Z", "fields": {"name": "gummies 200mg", "weight": "heavy"}}, "#,
        r#"{"fields": {"name": "vape", "description": null}}]}"#
      )
      .to_owned(),
      history("h2", r#""label": "unsure", "truncated": true, "#, &["a", "b"]),
    ]
    .join("\n");
    let parsed = parse_histories(&text, &schema(), None).unwrap();
    assert_eq!(parsed.histories.len(), 2);
    let first = &parsed.histories[0];
    assert_eq!((first.id.as_str(), first.brand_id.as_deref()), ("h1", Some("b1")));
    assert_eq!((first.label, first.recycled_at), (Some(Label::Recycle), Some(2)));
    assert_eq!(first.label_confidence, Some(Confidence::High));
    assert_eq!(first.original.as_ref().unwrap().texts["name"], "gummies");
    assert_eq!(first.versions.len(), 3);
    assert_eq!(first.versions[0].fields.quantities.len(), 1);
    assert_eq!(first.versions[1].at.as_deref(), Some("2026-02-01T00:00:00Z"));
    assert!(
      !first.versions[2].fields.texts.contains_key("description"),
      "a null is no value"
    );
    // What the schema cannot compare is counted, wherever in the history it was.
    assert_eq!(parsed.input.ignored[&("sku".to_owned(), Ignored::NotInSchema)], 1);
    assert_eq!(parsed.input.ignored[&("strain".to_owned(), Ignored::NotInSchema)], 1);
    assert_eq!(parsed.input.ignored[&("weight".to_owned(), Ignored::WrongKind)], 1);

    let second = &parsed.histories[1];
    assert_eq!(
      (second.label, second.label_text.as_str(), second.truncated),
      (None, "unsure", true)
    );
    assert_eq!(parsed.input.sha256.len(), 64);
  }

  #[test]
  fn a_line_that_is_not_a_history_is_refused_and_says_which_and_why() {
    let cases = [
      (history("h", r#""label": "edit", "recyled_at": 1, "#, &["a", "b"]), "unknown field `recyled_at`"),
      (history("h", r#""label": "edit", "#, &["a"]), "at least two"),
      (history("h", r#""label": "edit", "recycled_at": 1, "#, &["a", "b"]), "only a recycle"),
      (history("h", r#""label": "recycle", "recycled_at": 0, "#, &["a", "b"]), "after the first"),
      (history("h", r#""label": "recycle", "recycled_at": 2, "#, &["a", "b"]), "from 1 to 1"),
      (history("h", r#""label": "edit", "label_confidence": "certain", "#, &["a", "b"]), "not a confidence"),
      (
        r#"{"product": "h", "label": "edit", "versions": [{"fields": {"name": "a"}}, {"fields": {"weight": {"value": 1}}}]}"#
          .to_owned(),
        "version 1 of `h`",
      ),
    ];
    for (text, expected) in cases {
      let error = format!("{:#}", parse_histories(&text, &schema(), None).unwrap_err());
      assert!(
        error.contains("line 1") && error.contains(expected),
        "{expected}: {error}"
      );
    }
    let twice = [
      history("h", r#""label": "edit", "#, &["a", "b"]),
      history("h", r#""label": "edit", "#, &["c", "d"]),
    ];
    let error = parse_histories(&twice.join("\n"), &schema(), None)
      .unwrap_err()
      .to_string();
    assert!(error.contains("lines 1 and 2"), "{error}");
  }

  #[test]
  fn a_labeled_history_below_the_confidence_asked_for_is_left_out_and_counted() {
    let text = [
      history("low", r#""label": "edit", "label_confidence": "low", "#, &["a", "b"]),
      history("high", r#""label": "edit", "label_confidence": "high", "#, &["a", "b"]),
      history("none", r#""label": "edit", "#, &["a", "b"]),
      history(
        "unsure",
        r#""label": "unsure", "label_confidence": "low", "#,
        &["a", "b"],
      ),
    ]
    .join("\n");
    let parsed = parse_histories(&text, &schema(), Some(Confidence::Medium)).unwrap();
    let kept: Vec<&str> = parsed.histories.iter().map(|history| history.id.as_str()).collect();
    assert_eq!(kept, ["high", "none", "unsure"]);
    assert_eq!(parsed.input.skipped_confidence, 1);
  }
}
