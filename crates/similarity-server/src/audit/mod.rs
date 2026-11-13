//! Checking the store against a catalog: `record-similarity audit`.
//!
//! The service's baselines and the caller's own database have to agree, and they part without either noticing: a
//! save that failed after the service had accepted the edit, a write that was finished for a caller that had gone
//! (D38), a restore from backup on either side. A baseline that is stale means the next edit is judged against a
//! record that no longer exists. An audit reads an export of the catalog, in the format an import reads, lists
//! what the store holds, and says where the two disagree.
//!
//! It embeds nothing. The store keeps a fingerprint of each field's text, and the export's text is fingerprinted
//! the same way ([`similarity_core::PreparedRecord::hashes`]); quantities and brands are compared as they are.
//! What it cannot see is the anchor, which is a vector and has no fingerprints: an export's `anchor` is checked
//! as an import would check it and compared with nothing.
//!
//! It reports and repairs nothing, and is written against [`Scan`], which cannot write. A record the store lacks
//! is created by running the import on the same export (D20). A record that differs is the caller's to decide:
//! replacing a baseline is a forced write, which resets its anchor.
//!
//! A scan is not a snapshot, and neither is an export. A record edited between the two differs without anything
//! being wrong, which is what [`Options::exported_at`] is for; a record created or deleted while the store is
//! being listed may be reported as missing on one side. Before acting on a finding, audit again.

use std::collections::hash_map::Entry as Slot;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

use anyhow::Context;
use serde::Deserialize;
use similarity_core::Schema;
use similarity_store::{Entry, Read, Scan};

use crate::import::{self, ImportLine};

pub mod compare;
pub mod report;

pub use compare::{Difference, Expected, Finding, Kind, Summary, compare};

/// How an audit is run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
  /// How many points are read from the store at a time.
  pub page_records: usize,
  /// When the export was taken, in seconds since the Unix epoch, if that is known. A record written at or after
  /// that time is still reported where it differs, and is not counted as a difference
  /// ([`Summary::differences`]).
  pub exported_at: Option<u64>,
}

/// The id of a line that is not a record to import, where the line is at least JSON with an `id`. A line that is
/// refused for a key it should not have still says which record it is about, and that record is in the export.
fn id_of(line: &str) -> Option<String> {
  #[derive(Deserialize)]
  struct Named {
    id: String,
  }
  serde_json::from_str::<Named>(line)
    .ok()
    .map(|named| named.id)
    .filter(|id| !id.trim().is_empty())
}

/// One line of the export as an import would take it: the line and what it would be stored as, or the id (where
/// there is one) and the reason an import would refuse it.
fn read(schema: &Schema, line: std::io::Result<String>) -> Result<(ImportLine, Expected), (Option<String>, String)> {
  let text = line.map_err(|error| (None, format!("cannot read the line: {error}")))?;
  let parsed = import::parse(Ok(text.clone())).map_err(|error| (id_of(&text), format!("{error:#}")))?;
  match import::prepared(schema, &parsed) {
    Ok(record) => {
      let expected = Expected {
        line: 0,
        brand_id: parsed.brand_id.clone(),
        hashes: record.hashes(schema),
        quantities: record.quantities().clone(),
      };
      Ok((parsed, expected))
    }
    Err(reason) => Err((Some(parsed.id).filter(|id| !id.trim().is_empty()), reason)),
  }
}

/// An audit as it goes: what the export said, less what has been found in the store so far, and the counts.
struct Audit<F> {
  summary: Summary,
  /// What the export says should be stored: for each id, the first line an import would have taken. That is the
  /// line an import stored, since it skips an id it has already created. A record leaves when the store lists it.
  expected: HashMap<String, Expected>,
  /// The records the export names only on lines an import would refuse. They are in the export, and must not be
  /// reported as missing from it.
  invalid: HashSet<String>,
  exported_at: Option<u64>,
  found: F,
}

impl<F: FnMut(&Finding)> Audit<F> {
  fn report(&mut self, finding: &Finding) {
    self.summary.findings = self.summary.findings.saturating_add(1);
    (self.found)(finding);
  }

  /// Takes one line of the export.
  fn line(&mut self, schema: &Schema, number: usize, line: std::io::Result<String>) {
    self.summary.lines = self.summary.lines.saturating_add(1);
    match read(schema, line) {
      Ok((line, record)) => match self.expected.entry(line.id) {
        Slot::Vacant(slot) => {
          self.summary.export_records = self.summary.export_records.saturating_add(1);
          slot.insert(Expected { line: number, ..record });
        }
        Slot::Occupied(first) => {
          self.summary.duplicate_ids = self.summary.duplicate_ids.saturating_add(1);
          let finding = Finding {
            id: Some(first.key().clone()),
            line: Some(number),
            first_line: Some(first.get().line),
            brand_id: line.brand_id,
            ..Finding::new(Kind::DuplicateId)
          };
          self.report(&finding);
        }
      },
      Err((id, reason)) => {
        self.summary.invalid_lines = self.summary.invalid_lines.saturating_add(1);
        match &id {
          Some(id) => {
            self.invalid.insert(id.clone());
          }
          // A line that does not say which record it is can be matched with nothing.
          None => self.summary.not_compared = self.summary.not_compared.saturating_add(1),
        }
        self.report(&Finding {
          id,
          line: Some(number),
          reason: Some(reason),
          ..Finding::new(Kind::InvalidLine)
        });
      }
    }
  }

  /// Takes one thing the store holds.
  fn entry(&mut self, entry: Entry) {
    self.summary.points = self.summary.points.saturating_add(1);
    match entry {
      Entry::Record(stored) => {
        self.summary.stored_records = self.summary.stored_records.saturating_add(1);
        if let Some(given) = self.expected.remove(&stored.id) {
          let findings = compare(&given, &stored, self.exported_at);
          self.summary.compared(&findings);
          for finding in findings {
            self.report(&finding);
          }
        } else if self.invalid.remove(&stored.id) {
          self.summary.not_compared = self.summary.not_compared.saturating_add(1);
        } else {
          let finding = Finding::stored(Kind::NotInExport, &stored, self.exported_at);
          self.summary.not_in_export = self.summary.not_in_export.saturating_add(1);
          // A record created since the export was taken is not in it, and nothing is wrong.
          self.summary.after_export = self
            .summary
            .after_export
            .saturating_add(usize::from(finding.after_export == Some(true)));
          self.report(&finding);
        }
      }
      Entry::OtherLayout { id, layout } => {
        self.summary.other_layout = self.summary.other_layout.saturating_add(1);
        let records = self.summary.other_layouts.entry(layout.clone()).or_default();
        *records = records.saturating_add(1);
        self.uncompared(Kind::OtherLayout, Some(id), |finding| finding.layout = Some(layout));
      }
      Entry::Unreadable { id, reason } => {
        self.summary.unreadable = self.summary.unreadable.saturating_add(1);
        // An id here is a stored record that cannot be read: it is stored, and is not reported as missing.
        self.uncompared(Kind::Unreadable, id, |finding| finding.reason = Some(reason));
      }
    }
  }

  /// Reports something in the store that cannot be compared with what the export says of that record, if it says
  /// anything: the record is neither compared nor, later, reported as missing.
  fn uncompared(&mut self, kind: Kind, id: Option<String>, describe: impl FnOnce(&mut Finding)) {
    self.summary.not_compared = self.summary.not_compared.saturating_add(1);
    let given = id.as_ref().and_then(|id| {
      self.invalid.remove(id);
      self.expected.remove(id)
    });
    let mut finding = Finding {
      line: given.as_ref().map(|given| given.line),
      brand_id: given.and_then(|given| given.brand_id),
      id,
      ..Finding::new(kind)
    };
    describe(&mut finding);
    self.report(&finding);
  }

  /// Ends the audit once the whole store has been listed: only now is a record that was not seen a record that
  /// is not stored.
  fn finish(mut self) -> Summary {
    let mut missing: Vec<(String, Expected)> = std::mem::take(&mut self.expected).into_iter().collect();
    missing.sort_by_key(|(_, given)| given.line);
    for (id, given) in missing {
      self.summary.not_stored = self.summary.not_stored.saturating_add(1);
      self.report(&Finding {
        id: Some(id),
        line: Some(given.line),
        brand_id: given.brand_id,
        ..Finding::new(Kind::NotStored)
      });
    }
    // Named by the export only on lines that are invalid, and not stored either.
    self.summary.not_compared = self.summary.not_compared.saturating_add(self.invalid.len());
    self.summary
  }
}

/// Compares what `store` holds with what `export` says it should hold.
///
/// `found` is called for every finding as it is made: first what is wrong with the export itself, then what the
/// store holds, in the store's order, then the records the store lacks, in the export's. `progress` is called
/// after every page of the store with the counts so far.
///
/// The records of the export are held in memory until the store has been listed.
///
/// # Errors
///
/// Returns an error if the store cannot be listed to its end. Nothing is concluded from part of a store: every
/// record on a page that was not read would be reported as missing.
pub async fn run<S: Scan>(
  store: &S,
  schema: &Schema,
  export: impl BufRead,
  options: Options,
  found: impl FnMut(&Finding),
  mut progress: impl FnMut(&Summary),
) -> anyhow::Result<Summary> {
  let mut audit = Audit {
    summary: Summary::default(),
    expected: HashMap::new(),
    invalid: HashSet::new(),
    exported_at: options.exported_at,
    found,
  };
  for (number, line) in import::numbered(export) {
    audit.line(schema, number, line);
  }
  // An id is compared if any of its lines can be.
  let Audit { expected, invalid, .. } = &mut audit;
  invalid.retain(|id| !expected.contains_key(id));

  let mut from = None;
  loop {
    let page = store
      .scan(from.as_ref(), options.page_records.max(1), Read::Payload)
      .await
      .with_context(|| {
        format!(
          "the store could not be listed past its first {} point(s); nothing is concluded from part of a store",
          audit.summary.points
        )
      })?;
    for entry in page.entries {
      audit.entry(entry);
    }
    progress(&audit.summary);
    match page.next {
      Some(next) => from = Some(next),
      None => return Ok(audit.finish()),
    }
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use similarity_core::{FieldSpec, Policy, Quantity, QuantitySpec};
  use similarity_store::{Cursor, MemoryStore, Page, StoreError, VectorStore};

  use super::*;
  use crate::engine::{Engine, EvaluateInput, RecordInput, WriteMode, store_layout};
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

  type TestEngine = Engine<Arc<MemoryStore>, WordBagEmbedder>;

  /// An engine and the store it writes to, which holds whatever `before` put into it under another layout.
  async fn serving(before: &[&str]) -> (TestEngine, Arc<MemoryStore>) {
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let old = MemoryStore::new("another-model|category,name", dimension);
    for id in before {
      old
        .put(&similarity_store::StoredRecord {
          id: (*id).to_owned(),
          brand_id: None,
          vectors: similarity_core::RecordVectors {
            composite: vec![1.0; dimension],
            hashes: std::collections::BTreeMap::new(),
          },
          anchor: vec![1.0; dimension],
          quantities: similarity_core::Quantities::new(),
          anchor_quantities: similarity_core::Quantities::new(),
          revision: 1,
          updated_at: 1_700_000_000,
        })
        .await
        .unwrap();
    }
    let store = Arc::new(old.reopened(layout, dimension));
    let engine = Engine::new(Arc::clone(&store), embedder, schema(), Policy::new(0.75, None).unwrap());
    (engine, store)
  }

  async fn import(engine: &TestEngine, export: &str) -> import::Summary {
    let options = import::Options {
      batch_records: 2,
      stop_on_error: false,
    };
    import::run(engine, export.as_bytes(), options, |_| {}, |_| {})
      .await
      .unwrap()
  }

  async fn audit_with<S: Scan>(store: &S, export: &str, options: Options) -> (Summary, Vec<Finding>) {
    let mut findings = Vec::new();
    let summary = run(
      store,
      &schema(),
      export.as_bytes(),
      options,
      |finding| findings.push(finding.clone()),
      |_| {},
    )
    .await
    .unwrap();
    assert_eq!(summary.findings, findings.len());
    (summary, findings)
  }

  const PAGES_OF_TWO: Options = Options {
    page_records: 2,
    exported_at: None,
  };

  async fn audit<S: Scan>(store: &S, export: &str) -> (Summary, Vec<Finding>) {
    audit_with(store, export, PAGES_OF_TWO).await
  }

  fn told(findings: &[Finding]) -> Vec<(Kind, Option<&str>, Option<usize>)> {
    findings
      .iter()
      .map(|finding| (finding.kind, finding.id.as_deref(), finding.line))
      .collect()
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

  #[tokio::test]
  async fn an_export_that_was_just_imported_audits_clean() {
    let (engine, store) = serving(&[]).await;
    assert_eq!(import(&engine, EXPORT).await.imported, 3);

    let (summary, findings) = audit(&store, EXPORT).await;
    assert_eq!(findings, []);
    assert_eq!(
      summary,
      Summary {
        lines: 3,
        export_records: 3,
        points: 3,
        stored_records: 3,
        agree: 3,
        ..Summary::default()
      }
    );
    assert_eq!(summary.differences(), 0);

    // The same catalog written another way is the same catalog: other whitespace, another unit, other key order.
    let rewritten = concat!(
      r#"{"record": {"weight": {"min": 0.04, "max": 0.04, "unit": "kg"}, "category": " edibles ", "name": "blueberry  gummies\n"}, "brand_id": "brand-1", "id": "sku-1"}"#,
      "\n",
      r#"{"id": "sku-3", "brand_id": "brand-1", "record": {"name": "og kush pre-roll", "category": "  "}}"#,
      "\n",
      r#"{"id": "sku-2", "record": {"name": "sour diesel vape cartridge"}}"#,
      "\n",
    );
    let (summary, findings) = audit(&store, rewritten).await;
    assert_eq!((findings, summary.agree), (Vec::new(), 3));
  }

  #[tokio::test]
  async fn a_record_the_store_lacks_and_one_the_export_lacks_are_each_reported_once() {
    let (engine, store) = serving(&[]).await;
    import(&engine, EXPORT).await;
    engine.delete("sku-2".to_owned(), None).await.unwrap();
    import(
      &engine,
      r#"{"id": "sku-9", "brand_id": "brand-2", "record": {"name": "live resin"}}"#,
    )
    .await;

    for page_records in [1, 2, 3, 100] {
      let options = Options {
        page_records,
        exported_at: None,
      };
      let (summary, findings) = audit_with(&store, EXPORT, options).await;
      let mut kinds = told(&findings);
      kinds.sort();
      assert_eq!(
        kinds,
        [
          (Kind::NotStored, Some("sku-2"), Some(4)),
          (Kind::NotInExport, Some("sku-9"), None),
        ],
        "pages of {page_records}"
      );
      assert_eq!(
        summary,
        Summary {
          lines: 3,
          export_records: 3,
          points: 3,
          stored_records: 3,
          agree: 2,
          not_stored: 1,
          not_in_export: 1,
          findings: 2,
          ..Summary::default()
        }
      );
      assert_eq!(summary.differences(), 2);
    }
    // What the store says of the record the export lacks, and what the export says of the one the store lacks.
    let (_, findings) = audit(&store, EXPORT).await;
    let extra = findings
      .iter()
      .find(|finding| finding.kind == Kind::NotInExport)
      .unwrap();
    assert_eq!(
      (extra.stored_brand_id.as_deref(), extra.revision),
      (Some("brand-2"), Some(1))
    );
    let lacking = findings.last().unwrap();
    assert_eq!((lacking.kind, lacking.brand_id.as_deref()), (Kind::NotStored, None));
  }

  #[tokio::test]
  async fn a_stored_record_that_was_edited_differs_in_what_was_edited() {
    let (engine, store) = serving(&[]).await;
    import(&engine, EXPORT).await;
    let edit = EvaluateInput {
      record: RecordInput {
        id: Some("sku-1".to_owned()),
        brand_id: Some("brand-1".to_owned()),
        fields: vec![("name".to_owned(), "peach gummies".to_owned())],
        quantities: vec![("weight".to_owned(), Quantity::single(80.0, "g"))],
      },
      mode: WriteMode::Force,
      threshold: None,
    };
    engine.evaluate(edit).await.unwrap();

    let (summary, findings) = audit(&store, EXPORT).await;
    let differing: Vec<String> = findings
      .iter()
      .map(|finding| {
        assert_eq!(
          (finding.id.as_deref(), finding.line, finding.revision),
          (Some("sku-1"), Some(2), Some(2))
        );
        format!(
          "{} {} {}",
          finding.kind.code(),
          finding.field.as_deref().or(finding.quantity.as_deref()).unwrap(),
          finding.difference.unwrap().code()
        )
      })
      .collect();
    assert_eq!(
      differing,
      [
        "text_differs category only_in_export",
        "text_differs name differs",
        "quantity_differs weight differs",
      ]
    );
    assert_eq!(
      (
        summary.agree,
        summary.differ,
        summary.text_differs,
        summary.quantity_differs
      ),
      (2, 1, 1, 1)
    );
  }

  #[tokio::test]
  async fn a_record_written_since_the_export_is_flagged_and_is_not_a_difference() {
    let (engine, store) = serving(&[]).await;
    import(&engine, EXPORT).await;
    engine
      .delete("sku-3".to_owned(), Some("brand-1".to_owned()))
      .await
      .unwrap();
    import(
      &engine,
      r#"{"id": "sku-3", "brand_id": "brand-1", "record": {"name": "og kush pre-roll two pack"}}"#,
    )
    .await;
    import(&engine, r#"{"id": "sku-9", "record": {"name": "live resin"}}"#).await;
    let times = [
      store.get("sku-3").await.unwrap().unwrap().updated_at,
      store.get("sku-9").await.unwrap().unwrap().updated_at,
    ];
    let (first, last) = (times[0].min(times[1]), times[0].max(times[1]));

    // Taken before those writes: both are edits made since, as far as anyone can tell. So is one taken in the
    // second of the first of them.
    let before = Options {
      page_records: 2,
      exported_at: Some(first),
    };
    let (summary, findings) = audit_with(&store, EXPORT, before).await;
    assert_eq!(findings.len(), 2);
    assert!(findings.iter().all(|finding| finding.after_export == Some(true)));
    assert_eq!((summary.differ, summary.not_in_export, summary.after_export), (1, 1, 2));
    assert_eq!(summary.differences(), 0);

    // Taken after them: the store was not written since, so the export is wrong or the store is.
    let after = Options {
      page_records: 2,
      exported_at: Some(last.saturating_add(1)),
    };
    let (summary, findings) = audit_with(&store, EXPORT, after).await;
    assert!(findings.iter().all(|finding| finding.after_export == Some(false)));
    assert_eq!((summary.after_export, summary.differences()), (0, 2));

    // And with no time given nothing is set aside, and nothing is said about it.
    let (summary, findings) = audit(&store, EXPORT).await;
    assert!(findings.iter().all(|finding| finding.after_export.is_none()));
    assert_eq!(summary.differences(), 2);
  }

  #[tokio::test]
  async fn an_invalid_line_is_reported_and_its_id_is_not_called_missing_from_the_export() {
    let (engine, store) = serving(&[]).await;
    import(&engine, EXPORT).await;
    let flawed = concat!(
      r#"{"id": "sku-1", "brand_id": "brand-1", "record": {"name": "blueberry gummies", "colour": "blue"}}"#,
      "\n",
      r#"{"id": "sku-2", "tenant": "brand-1", "record": {"name": "sour diesel vape cartridge"}}"#,
      "\n",
      "not json at all\n",
      r#"{"id": "sku-7", "record": {"category": "edibles"}}"#,
      "\n",
      r#"{"id": "sku-3", "brand_id": "brand-1", "record": {"name": "og kush pre-roll"}}"#,
      "\n",
    );

    let (summary, findings) = audit(&store, flawed).await;
    assert_eq!(
      told(&findings),
      [
        (Kind::InvalidLine, Some("sku-1"), Some(1)),
        (Kind::InvalidLine, Some("sku-2"), Some(2)),
        (Kind::InvalidLine, None, Some(3)),
        (Kind::InvalidLine, Some("sku-7"), Some(4)),
      ],
      "and no record is said to be missing from an export that names it"
    );
    assert!(findings[0].reason.as_deref().unwrap().contains("colour"));
    assert!(findings[1].reason.as_deref().unwrap().contains("tenant"));
    assert_eq!(
      summary,
      Summary {
        lines: 5,
        invalid_lines: 4,
        export_records: 1,
        points: 3,
        stored_records: 3,
        agree: 1,
        // The two stored records the export says nothing usable about, the line that names no record, and the
        // record that is neither valid in the export nor stored.
        not_compared: 4,
        findings: 4,
        ..Summary::default()
      }
    );
    assert_eq!(
      summary.differences(),
      4,
      "an audit vouches for nothing it could not compare"
    );
  }

  #[tokio::test]
  async fn the_first_line_an_import_would_have_stored_is_compared_and_later_ones_are_duplicates() {
    let twice = concat!(
      r#"{"id": "sku-1", "record": {"category": "no name, so an import refuses this line"}}"#,
      "\n",
      r#"{"id": "sku-1", "brand_id": "brand-1", "record": {"name": "blueberry gummies"}}"#,
      "\n",
      r#"{"id": "sku-1", "brand_id": "brand-1", "record": {"name": "peach gummies"}}"#,
      "\n",
    );
    let (engine, store) = serving(&[]).await;
    let imported = import(&engine, twice).await;
    assert_eq!((imported.imported, imported.skipped, imported.failed), (1, 1, 1));

    let (summary, findings) = audit(&store, twice).await;
    assert_eq!(
      told(&findings),
      [
        (Kind::InvalidLine, Some("sku-1"), Some(1)),
        (Kind::DuplicateId, Some("sku-1"), Some(3)),
      ]
    );
    assert_eq!(findings[1].first_line, Some(2));
    assert_eq!(
      summary,
      Summary {
        lines: 3,
        invalid_lines: 1,
        duplicate_ids: 1,
        export_records: 1,
        points: 1,
        stored_records: 1,
        agree: 1,
        findings: 2,
        ..Summary::default()
      }
    );
    assert_eq!(
      summary.differences(),
      0,
      "what the import stored is what the export's first usable line says"
    );
  }

  #[tokio::test]
  async fn a_record_under_another_layout_is_reported_and_not_compared() {
    let (engine, store) = serving(&["sku-2", "old-1"]).await;
    // The import cannot create `sku-2`: something is stored under that id, and this server cannot read it.
    let imported = import(&engine, EXPORT).await;
    assert_eq!((imported.imported, imported.failed), (2, 1));

    let (summary, findings) = audit(&store, EXPORT).await;
    let mut other = told(&findings);
    other.sort();
    assert_eq!(
      other,
      [
        (Kind::OtherLayout, Some("old-1"), None),
        (Kind::OtherLayout, Some("sku-2"), Some(4)),
      ],
      "neither is said to be missing from the store or from the export"
    );
    assert!(
      findings
        .iter()
        .all(|finding| finding.layout.as_deref() == Some("another-model|category,name"))
    );
    assert_eq!(
      summary,
      Summary {
        lines: 3,
        export_records: 3,
        points: 4,
        stored_records: 2,
        other_layout: 2,
        other_layouts: [("another-model|category,name".to_owned(), 2)].into(),
        agree: 2,
        not_compared: 2,
        findings: 2,
        ..Summary::default()
      }
    );
  }

  /// A store whose first page can be read and whose second cannot.
  struct Failing(MemoryStore);

  impl Scan for Failing {
    async fn scan(&self, from: Option<&Cursor>, limit: usize, read: Read) -> similarity_store::Result<Page> {
      match from {
        None => self.0.scan(None, limit, read).await,
        Some(_) => Err(StoreError::Unavailable("connection reset".to_owned())),
      }
    }
  }

  #[tokio::test]
  async fn a_store_that_cannot_be_listed_to_its_end_ends_the_audit_with_no_summary() {
    let (engine, store) = serving(&[]).await;
    import(&engine, EXPORT).await;
    drop(engine);
    let failing = Failing(Arc::into_inner(store).unwrap());

    let mut findings = Vec::new();
    let outcome = run(
      &failing,
      &schema(),
      EXPORT.as_bytes(),
      PAGES_OF_TWO,
      |finding| findings.push(finding.clone()),
      |_| {},
    )
    .await;
    let error = format!("{:#}", outcome.unwrap_err());
    assert!(
      error.contains("first 2 point(s)") && error.contains("connection reset"),
      "{error}"
    );
    assert!(
      findings.iter().all(|finding| finding.kind != Kind::NotStored),
      "the record on the page that was not read is not called missing: {findings:?}"
    );

    // With pages large enough to hold the store there is no second page to fail on.
    let whole = Options {
      page_records: 3,
      exported_at: None,
    };
    assert_eq!(audit_with(&failing, EXPORT, whole).await.0.agree, 3);
  }

  #[tokio::test]
  async fn progress_is_told_after_every_page_of_the_store() {
    let (engine, store) = serving(&[]).await;
    import(&engine, EXPORT).await;
    let mut seen = Vec::new();
    run(
      &store,
      &schema(),
      EXPORT.as_bytes(),
      PAGES_OF_TWO,
      |_| {},
      |summary| seen.push(summary.points),
    )
    .await
    .unwrap();
    assert_eq!(seen, [2, 3]);
  }
}
