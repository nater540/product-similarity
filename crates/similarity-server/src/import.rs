//! Loading a whole catalog: `record-similarity import`.
//!
//! The gRPC API takes one record at a time and judges it. A catalog that is being loaded for the first time, or
//! loaded again into a new collection after a change of model (D12), is neither judged nor in a hurry record by
//! record: every line becomes a baseline, and what matters is getting through tens of thousands of them. So this
//! reads an export, hands the engine a batch at a time (one call to the model per batch), and writes straight to
//! the store.
//!
//! One line of the export is one record:
//!
//! ```json
//! {"id": "sku-1", "brand_id": "brand-7", "record": {"name": "...", "weight_per_item": {"value": 3.5, "unit": "g"}},
//!  "anchor": {"name": "..."}}
//! ```
//!
//! `record` is the record as it is now, in the shape every other tool reads (a string is a text field, an object is
//! a quantity). `anchor`, optional, is the record as it was first created. With it, drift is measured from where
//! the record really began and not from the day of the import (see [`Engine::import`]).
//!
//! An import only creates. An id that is already stored is skipped, which is what makes a run that was interrupted
//! safe to simply run again.

use std::io::BufRead;

use anyhow::Context;
use serde::Deserialize;
use similarity_core::{PreparedRecord, Schema};
use similarity_store::VectorStore;

use crate::engine::{Engine, ImportInput, Imported, Original, RecordInput};
use crate::offline::RecordFields;
use crate::worker::TextEmbedder;

/// One line of an export.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportLine {
  /// The record's id.
  pub id: String,
  /// The key of the brand that owns the record, if records have brands.
  #[serde(default)]
  pub brand_id: Option<String>,
  /// The record as it is now.
  pub record: RecordFields,
  /// The record as it was first created, if that is still known.
  #[serde(default)]
  pub anchor: Option<RecordFields>,
}

impl From<ImportLine> for ImportInput {
  fn from(line: ImportLine) -> Self {
    Self {
      record: RecordInput {
        id: Some(line.id),
        brand_id: line.brand_id,
        fields: line.record.texts.into_iter().collect(),
        quantities: line.record.quantities.into_iter().collect(),
      },
      anchor: line.anchor.map(|anchor| Original {
        fields: anchor.texts.into_iter().collect(),
        quantities: anchor.quantities.into_iter().collect(),
      }),
    }
  }
}

/// How an import is run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
  /// How many records go to the engine, and so to the model, at a time.
  pub batch_records: usize,
  /// Stop at the first record that cannot be imported, instead of counting it and going on.
  pub stop_on_error: bool,
}

/// What an import did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Summary {
  /// Lines of the export that held a record.
  pub records: usize,
  /// Records that were not stored and now are.
  pub imported: usize,
  /// Records whose id was already stored. They were left as they were.
  pub skipped: usize,
  /// Records that could not be imported: not valid JSON, not fitting the schema, or stored under another brand.
  pub failed: usize,
}

/// A record that was not imported, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
  /// The line of the export it is on.
  pub line: usize,
  /// Its id, if the line got as far as having one.
  pub id: Option<String>,
  /// What is wrong with it.
  pub reason: String,
}

/// The lines of an export that hold a record, parsed, with their one-based line numbers. Blank lines and lines
/// starting with `#` are passed over, as in every other file these tools read.
pub(crate) fn lines(reader: impl BufRead) -> impl Iterator<Item = (usize, anyhow::Result<ImportLine>)> {
  reader
    .lines()
    .enumerate()
    .map(|(index, line)| (index.saturating_add(1), line))
    .filter(|(_, line)| {
      line.as_ref().map_or(true, |line| {
        !line.trim().is_empty() && !line.trim_start().starts_with('#')
      })
    })
    .map(|(number, line)| {
      let parsed = line
        .context("cannot read the line")
        .and_then(|line| serde_json::from_str::<ImportLine>(&line).context("not a record to import"));
      (number, parsed)
    })
}

/// Imports every record of `export` that is not stored yet.
///
/// `report` is called for every record that could not be imported, as it is found, and `progress` after every
/// batch with the totals so far.
///
/// # Errors
///
/// Returns an error if the model fails (nothing of that batch is written) or, with
/// [`Options::stop_on_error`], at the first record that cannot be imported. Records imported before the error stay
/// imported, and running the import again skips them.
pub async fn run<S: VectorStore, E: TextEmbedder>(
  engine: &Engine<S, E>,
  export: impl BufRead,
  options: Options,
  mut report: impl FnMut(&Failure),
  mut progress: impl FnMut(&Summary),
) -> anyhow::Result<Summary> {
  let mut summary = Summary::default();
  let mut fail = |summary: &mut Summary, failure: Failure| -> anyhow::Result<()> {
    summary.failed = summary.failed.saturating_add(1);
    report(&failure);
    if options.stop_on_error {
      anyhow::bail!("line {}: {}", failure.line, failure.reason);
    }
    Ok(())
  };

  let mut batch: Vec<(usize, ImportLine)> = Vec::new();
  let mut parsed = lines(export).peekable();
  while let Some((number, line)) = parsed.next() {
    summary.records = summary.records.saturating_add(1);
    match line {
      Ok(line) => batch.push((number, line)),
      Err(error) => fail(
        &mut summary,
        Failure {
          line: number,
          id: None,
          reason: format!("{error:#}"),
        },
      )?,
    }
    if batch.len() < options.batch_records.max(1) && parsed.peek().is_some() {
      continue;
    }

    let (numbers, ids): (Vec<usize>, Vec<String>) =
      batch.iter().map(|(number, line)| (*number, line.id.clone())).unzip();
    let inputs = batch.drain(..).map(|(_, line)| ImportInput::from(line)).collect();
    let outcomes = engine
      .import(inputs)
      .await
      .with_context(|| format!("the model failed on the records up to line {number}; none of that batch was stored"))?;
    for ((line, id), outcome) in numbers.into_iter().zip(ids).zip(outcomes) {
      match outcome {
        Ok(Imported::Created) => summary.imported = summary.imported.saturating_add(1),
        Ok(Imported::Skipped) => summary.skipped = summary.skipped.saturating_add(1),
        Err(error) => fail(
          &mut summary,
          Failure {
            line,
            id: Some(id),
            reason: error.to_string(),
          },
        )?,
      }
    }
    progress(&summary);
  }
  Ok(summary)
}

/// A line's record as an import would prepare it, or the reason an import would refuse the line, as far as the
/// schema can tell. The one place that says what a valid line is for everything that reads an export without a
/// model: [`check`] and the audit.
pub(crate) fn prepared(schema: &Schema, line: &ImportLine) -> Result<PreparedRecord, String> {
  if line.id.trim().is_empty() {
    return Err("`id` is blank".to_owned());
  }
  if line.brand_id.as_deref().is_some_and(|brand| brand.trim().is_empty()) {
    return Err("`brand_id` is present but blank; omit it or give it a value".to_owned());
  }
  let validate = |record: &RecordFields| -> similarity_core::Result<PreparedRecord> {
    let quantities = record
      .quantities
      .iter()
      .map(|(name, quantity)| (name, quantity.clone()));
    PreparedRecord::new(schema, &record.texts)?.with_quantities(schema, quantities)
  };
  let record = validate(&line.record).map_err(|error| error.to_string())?;
  if let Some(anchor) = &line.anchor {
    validate(anchor).map_err(|error| format!("the original version: {error}"))?;
  }
  Ok(record)
}

/// Reads an export and checks every record against `schema`, without a model and without a store.
///
/// This is the quick way to find out whether an export will import: a misnamed field, a quantity sent as text, a
/// missing required field, a blank id or brand, on every line that has one. `report` is called for each. Records
/// are not embedded, so only what the schema can tell is checked; whether an id is already stored is not.
pub fn check(schema: &Schema, export: impl BufRead, mut report: impl FnMut(&Failure)) -> Summary {
  let mut summary = Summary::default();
  for (number, line) in lines(export) {
    summary.records = summary.records.saturating_add(1);
    let problem = match line {
      Err(error) => Some((None, format!("{error:#}"))),
      Ok(line) => prepared(schema, &line).err().map(|reason| {
        let id = Some(line.id).filter(|id| !id.trim().is_empty());
        (id, reason)
      }),
    };
    if let Some((id, reason)) = problem {
      summary.failed = summary.failed.saturating_add(1);
      report(&Failure {
        line: number,
        id,
        reason,
      });
    }
  }
  summary
}

#[cfg(test)]
mod tests {
  use similarity_core::{FieldSpec, Policy, QuantitySpec};
  use similarity_store::MemoryStore;

  use super::*;
  use crate::engine::store_layout;
  use crate::worker::testing::WordBagEmbedder;

  fn schema() -> Schema {
    let field = |name: &str, weight: f32, required: bool| FieldSpec {
      name: name.to_owned(),
      weight,
      required,
      reject_below: None,
    };
    Schema::new(vec![field("name", 4.0, true), field("category", 2.0, false)])
      .unwrap()
      .with_quantities(vec![QuantitySpec {
        name: "weight".to_owned(),
        reject_above_factor: None,
        reject_incomparable: false,
      }])
      .unwrap()
  }

  fn engine() -> Engine<MemoryStore, WordBagEmbedder> {
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    Engine::new(
      MemoryStore::new(layout, dimension),
      embedder,
      schema(),
      Policy::new(0.75, None).unwrap(),
    )
  }

  const EXPORT: &str = concat!(
    "# three records and a comment\n",
    r#"{"id": "sku-1", "brand_id": "brand-1", "record": {"name": "blueberry gummies", "category": "edibles", "weight": {"value": 40, "unit": "g"}}}"#,
    "\n\n",
    r#"{"id": "sku-2", "record": {"name": "sour diesel vape cartridge"}, "anchor": {"name": "sour diesel cartridge", "weight": {"value": 1, "unit": "g"}}}"#,
    "\n",
    r#"{"id": "sku-3", "brand_id": "brand-1", "record": {"name": "og kush pre-roll"}}"#,
    "\n",
  );

  async fn import(
    engine: &Engine<MemoryStore, WordBagEmbedder>,
    export: &str,
    options: Options,
  ) -> (anyhow::Result<Summary>, Vec<Failure>, usize) {
    let (mut failures, mut batches) = (Vec::new(), 0);
    let outcome = run(
      engine,
      export.as_bytes(),
      options,
      |failure| failures.push(failure.clone()),
      |_| batches += 1,
    )
    .await;
    (outcome, failures, batches)
  }

  const TWO_AT_A_TIME: Options = Options {
    batch_records: 2,
    stop_on_error: false,
  };

  #[tokio::test]
  async fn an_export_is_imported_in_batches_and_a_second_run_imports_nothing() {
    let engine = engine();
    let (summary, failures, batches) = import(&engine, EXPORT, TWO_AT_A_TIME).await;
    assert_eq!(
      summary.unwrap(),
      Summary {
        records: 3,
        imported: 3,
        skipped: 0,
        failed: 0
      }
    );
    assert!(failures.is_empty());
    assert_eq!(batches, 2, "two records, then the one that was left");

    let first = engine.store().get("sku-1").await.unwrap().unwrap();
    assert_eq!((first.revision, first.brand_id.as_deref()), (1, Some("brand-1")));
    assert_eq!(first.quantities["weight"], similarity_core::Quantity::single(40.0, "g"));
    let second = engine.store().get("sku-2").await.unwrap().unwrap();
    assert_ne!(
      second.anchor, second.vectors.composite,
      "its original version is its anchor"
    );
    assert!(second.quantities.is_empty());
    assert_eq!(
      second.anchor_quantities["weight"],
      similarity_core::Quantity::single(1.0, "g"),
      "a weight it had when it was created and has no longer"
    );

    // Interrupted or not, running it again is the way to resume: what is there is skipped.
    let before = engine.store().get("sku-1").await.unwrap();
    let (again, ..) = import(&engine, EXPORT, TWO_AT_A_TIME).await;
    assert_eq!(
      again.unwrap(),
      Summary {
        records: 3,
        imported: 0,
        skipped: 3,
        failed: 0
      }
    );
    assert_eq!(engine.store().get("sku-1").await.unwrap(), before);
  }

  const FLAWED: &str = concat!(
    r#"{"id": "ok-1", "record": {"name": "blueberry gummies"}}"#,
    "\n",
    "this line is not JSON\n",
    r#"{"id": "no-name", "record": {"category": "edibles"}}"#,
    "\n",
    r#"{"id": "typo", "record": {"name": "x", "categry": "edibles"}}"#,
    "\n",
    r#"{"id": "as-text", "record": {"name": "x", "weight": "40 g"}}"#,
    "\n",
    r#"{"id": "extra", "record": {"name": "x"}, "note": "not a key of a line"}"#,
    "\n",
    r#"{"id": "ok-2", "record": {"name": "og kush pre-roll"}, "anchor": {"category": "flower"}}"#,
    "\n",
    r#"{"id": "ok-3", "record": {"name": "og kush pre-roll"}}"#,
    "\n",
  );

  #[tokio::test]
  async fn records_that_cannot_be_imported_are_reported_by_line_and_the_rest_go_in() {
    let engine = engine();
    let (summary, failures, _) = import(&engine, FLAWED, TWO_AT_A_TIME).await;
    assert_eq!(
      summary.unwrap(),
      Summary {
        records: 8,
        imported: 2,
        skipped: 0,
        failed: 6
      }
    );
    let reported: Vec<(usize, Option<&str>)> = failures
      .iter()
      .map(|failure| (failure.line, failure.id.as_deref()))
      .collect();
    assert_eq!(
      reported,
      [
        (2, None),
        (3, Some("no-name")),
        (4, Some("typo")),
        (5, Some("as-text")),
        (6, None),
        (7, Some("ok-2")),
      ]
    );
    let reason = |line: usize| {
      failures
        .iter()
        .find(|failure| failure.line == line)
        .unwrap()
        .reason
        .as_str()
    };
    assert!(reason(2).contains("not a record to import"), "{}", reason(2));
    assert!(reason(3).contains("required field `name`"), "{}", reason(3));
    assert!(reason(4).contains("unknown field `categry`"), "{}", reason(4));
    assert!(reason(5).contains("quantity, not a text field"), "{}", reason(5));
    assert!(reason(6).contains("unknown field `note`"), "{}", reason(6));
    assert!(reason(7).contains("the original version"), "{}", reason(7));

    assert!(engine.store().get("ok-1").await.unwrap().is_some());
    assert!(engine.store().get("ok-3").await.unwrap().is_some());
    for id in ["no-name", "typo", "as-text", "extra", "ok-2"] {
      assert!(engine.store().get(id).await.unwrap().is_none(), "{id}");
    }
  }

  #[tokio::test]
  async fn an_export_with_nothing_importable_never_reaches_the_model() {
    let engine = engine();
    let nothing = "not json\n{\"id\": \"x\", \"record\": {\"category\": \"edibles\"}}\n";
    let (summary, failures, _) = import(&engine, nothing, TWO_AT_A_TIME).await;
    assert_eq!(
      summary.unwrap(),
      Summary {
        records: 2,
        imported: 0,
        skipped: 0,
        failed: 2
      }
    );
    assert_eq!(failures.len(), 2);

    let (empty, ..) = import(&engine, "# only a comment\n\n", TWO_AT_A_TIME).await;
    assert_eq!(empty.unwrap(), Summary::default());
  }

  #[tokio::test]
  async fn stopping_on_the_first_error_keeps_what_was_imported_before_it() {
    let engine = engine();
    let options = Options {
      batch_records: 1,
      stop_on_error: true,
    };
    let (outcome, failures, _) = import(&engine, FLAWED, options).await;
    let error = outcome.unwrap_err().to_string();
    assert!(error.contains("line 2"), "{error}");
    assert_eq!(failures.len(), 1);
    assert!(engine.store().get("ok-1").await.unwrap().is_some());
    assert!(engine.store().get("ok-3").await.unwrap().is_none(), "it stopped");
  }

  #[test]
  fn an_export_can_be_checked_against_the_schema_without_a_model_or_a_store() {
    let mut failures = Vec::new();
    let summary = check(&schema(), FLAWED.as_bytes(), |failure| failures.push(failure.clone()));
    assert_eq!(
      summary,
      Summary {
        records: 8,
        imported: 0,
        skipped: 0,
        failed: 6
      }
    );
    let lines: Vec<usize> = failures.iter().map(|failure| failure.line).collect();
    assert_eq!(lines, [2, 3, 4, 5, 6, 7], "the same lines an import would refuse");

    let mut none = Vec::new();
    let clean = check(&schema(), EXPORT.as_bytes(), |failure| none.push(failure.clone()));
    assert_eq!((clean.records, clean.failed, none.len()), (3, 0, 0));

    let blank = r#"{"id": "  ", "record": {"name": "x"}}"#;
    let summary = check(&schema(), blank.as_bytes(), |failure| failures.push(failure.clone()));
    assert_eq!(summary.failed, 1);
    assert!(failures.last().unwrap().reason.contains("blank"));
  }

  #[tokio::test]
  async fn a_blank_brand_is_refused_by_the_check_as_by_the_import() {
    let blank = r#"{"id": "sku-1", "brand_id": "  ", "record": {"name": "blueberry gummies"}}"#;

    let mut checked = Vec::new();
    let summary = check(&schema(), blank.as_bytes(), |failure| checked.push(failure.clone()));
    assert_eq!((summary.records, summary.failed), (1, 1));
    assert_eq!(checked[0].id.as_deref(), Some("sku-1"));
    assert!(
      checked[0].reason.contains("`brand_id` is present but blank"),
      "{}",
      checked[0].reason
    );

    let (outcome, imported, _) = import(&engine(), blank, TWO_AT_A_TIME).await;
    assert_eq!(outcome.unwrap().failed, 1, "the import refuses the same line");
    assert!(imported[0].reason.contains("brand_id"), "{}", imported[0].reason);
  }
}
