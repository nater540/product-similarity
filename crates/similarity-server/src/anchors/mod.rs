//! Keeping a copy of every record's anchor, and putting anchors back: `record-similarity anchors`.
//!
//! If a collection is lost, the caller's own database can give back every record as it is now: an import makes a
//! baseline of each. It cannot give back the anchors. An anchor is the first value each field of a record ever
//! had, the server keeps no text to make it from again, and the caller's database has forgotten what a record
//! said when it was created. A catalog loaded again without them has every record's drift history reset to the
//! day of the import, and says nothing about it: which is the state somebody walking a record away from what it
//! was would ask for.
//!
//! [`export`] writes every record's anchor to a file ([`file`]), read out of the store by a scan that changes
//! nothing. [`restore`] reads such a file and asks the engine, record by record, to put the anchor back
//! ([`Engine::restore_anchor`]), which it does only for a record that has no history of its own to lose and is
//! what the backup was taken of. Everything else is left as it is and reported.
//!
//! A backup belongs to the layout it was taken under. An anchor is a vector of one model over one set of fields
//! (D12), and after a change of either it means nothing: a restore refuses a backup of another layout whole.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};

use anyhow::Context;
use similarity_store::{Entry, Read, Scan, VectorStore};

use crate::engine::{Engine, EngineError, Restored};
use crate::worker::TextEmbedder;

pub mod file;

use file::{Header, Kept, Writer};

/// What a backup was made of.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Exported {
  /// Points the store holds, of any kind.
  pub points: usize,
  /// Records whose anchors are in the backup.
  pub records: usize,
  /// Records stored under another layout. Their anchors are another model's and are not kept.
  pub other_layout: usize,
  /// Points that could not be read as a record with an anchor. Nothing of them is kept.
  pub unreadable: usize,
}

/// Writes the anchor of every record `store` holds under its layout to `out`, as a backup that says it is of
/// `header`.
///
/// `skipped` is called for everything in the store that is not kept, with its id (where it has one) and why.
/// `progress` is called after every page of `page_records` points with the counts so far.
///
/// The store is listed while it may be written to, so the backup is of no single moment: each record is in it
/// once, with the anchor it had when its page was read. An anchor changes only when a record is created or a
/// write is forced, so that is as much of a moment as an anchor has.
///
/// # Errors
///
/// Returns an error if the store cannot be listed to its end or `out` cannot be written. What was written until
/// then has no last line, and is not a backup.
pub async fn export<S: Scan, W: Write>(
  store: &S,
  header: &Header,
  out: W,
  page_records: usize,
  mut skipped: impl FnMut(Option<&str>, &str),
  mut progress: impl FnMut(&Exported),
) -> anyhow::Result<(W, Exported)> {
  let mut writer = Writer::begin(out, header)?;
  let mut exported = Exported::default();
  let mut from = None;
  loop {
    let page = store
      .scan(from.as_ref(), page_records.max(1), Read::Anchor)
      .await
      .with_context(|| {
        format!(
          "the store could not be listed past its first {} point(s); what was written is not a backup",
          exported.points
        )
      })?;
    for entry in page.entries {
      exported.points = exported.points.saturating_add(1);
      match entry {
        Entry::Record(record) => {
          let id = record.id.clone();
          if let Some(kept) = Kept::of(record) {
            writer.anchor(kept)?;
            exported.records = exported.records.saturating_add(1);
          } else {
            // A store that was asked for anchors and gave none: counted, never passed over in silence.
            exported.unreadable = exported.unreadable.saturating_add(1);
            skipped(Some(&id), "the store listed the record without its anchor");
          }
        }
        Entry::OtherLayout { id, layout } => {
          exported.other_layout = exported.other_layout.saturating_add(1);
          skipped(Some(&id), &format!("it is stored under another layout, `{layout}`"));
        }
        Entry::Unreadable { id, reason } => {
          exported.unreadable = exported.unreadable.saturating_add(1);
          skipped(id.as_deref(), &reason);
        }
      }
    }
    progress(&exported);
    match page.next {
      Some(next) => from = Some(next),
      None => break,
    }
  }
  let (out, written) = writer.end()?;
  anyhow::ensure!(
    written == exported.records,
    "{written} record(s) were written and {} were counted",
    exported.records
  );
  Ok((out, exported))
}

/// What a restore did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Summary {
  /// Records the backup holds.
  pub records: usize,
  /// How many of them came to each outcome.
  pub outcomes: BTreeMap<Restored, usize>,
  /// Records the engine could not be asked about: the store did not answer, or holds the record under another
  /// layout.
  pub failed: usize,
}

impl Summary {
  /// How many records came to `outcome`.
  #[must_use]
  pub fn count(&self, outcome: Restored) -> usize {
    self.outcomes.get(&outcome).copied().unwrap_or_default()
  }
}

/// Puts back the anchors of a backup, or with `write` off says what putting them back would do.
///
/// The backup is read twice, which is why it is given as something that opens it: once from its first line to
/// its last, to know that it is whole and of this layout before anything is written, and once to restore.
/// `told` is called for every record with what came of it. `layout` and `dimension` are what the engine's store
/// was opened with.
///
/// # Errors
///
/// Returns an error, having written nothing, if the backup is not whole ([`file::check`]) or is of another
/// layout than this server's. A record the engine cannot be asked about fails by itself and is counted.
pub async fn restore<S, E, R>(
  engine: &Engine<S, E>,
  (layout, dimension): (&str, usize),
  open: impl Fn() -> anyhow::Result<R>,
  write: bool,
  mut told: impl FnMut(&str, &Result<Restored, EngineError>),
  mut progress: impl FnMut(&Summary),
) -> anyhow::Result<Summary>
where
  S: VectorStore,
  E: TextEmbedder,
  R: BufRead,
{
  let whole = file::check(open()?)?;
  anyhow::ensure!(
    whole.header.layout == layout && whole.header.dimension == dimension,
    "the backup is of layout `{}` ({} components), and this server's is `{layout}` ({dimension}): an anchor is a \
     vector of the model and fields it was made with, and means nothing under another",
    whole.header.layout,
    whole.header.dimension
  );

  let (_, records) = file::read(open()?)?;
  let mut summary = Summary::default();
  for kept in records {
    // The file was read whole a moment ago. If it cannot be now, it has changed underneath.
    let kept = kept.context("the backup changed while it was being restored")?;
    summary.records = summary.records.saturating_add(1);
    let outcome = engine.restore_anchor(&kept, write).await;
    match &outcome {
      Ok(restored) => {
        let count = summary.outcomes.entry(*restored).or_default();
        *count = count.saturating_add(1);
      }
      Err(_) => summary.failed = summary.failed.saturating_add(1),
    }
    told(&kept.id, &outcome);
    if summary.records.is_multiple_of(PROGRESS_EVERY) {
      progress(&summary);
    }
  }
  Ok(summary)
}

/// How many records a restore gets through between two reports of how far it is.
const PROGRESS_EVERY: usize = 1000;

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use similarity_core::{FieldSpec, Policy, Quantity, QuantitySpec, Schema};
  use similarity_store::{Cursor, MemoryStore, Page, StoreError};

  use super::*;
  use crate::engine::{EvaluateInput, ImportInput, Original, RecordInput, WriteMode, store_layout};
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

  /// An engine on an empty store, the store, and the layout it was opened with.
  fn serving() -> (TestEngine, Arc<MemoryStore>, (String, usize)) {
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema(), &embedder);
    let store = Arc::new(MemoryStore::new(layout.as_str(), dimension));
    let engine = Engine::new(
      Arc::clone(&store),
      embedder,
      schema(),
      Policy::new(0.75, Some(0.5)).unwrap(),
    );
    (engine, store, (layout, dimension))
  }

  fn header(layout: &(String, usize)) -> Header {
    Header {
      format: file::FORMAT,
      fingerprint: "word-bag-v1".to_owned(),
      layout: layout.0.clone(),
      dimension: layout.1,
      collection: "records".to_owned(),
      taken_at: 1_794_591_798,
    }
  }

  /// A record as it is now, and optionally the name it was created with and what it weighed then.
  fn product(id: &str, name: &str, created_as: Option<&str>) -> ImportInput {
    let record = |name: &str, weight: f64| RecordInput {
      id: Some(id.to_owned()),
      brand_id: Some("brand-1".to_owned()),
      fields: vec![
        ("name".to_owned(), name.to_owned()),
        ("category".to_owned(), "vape pens".to_owned()),
      ],
      quantities: vec![("weight".to_owned(), Quantity::single(weight, "g"))],
    };
    ImportInput {
      record: record(name, 2.0),
      anchor: created_as.map(|name| {
        let original = record(name, 1.0);
        Original {
          fields: original.fields,
          quantities: original.quantities,
        }
      }),
    }
  }

  /// The catalog: two products that have come a long way from what they were created as, and one that has not.
  fn catalog(with_originals: bool) -> Vec<ImportInput> {
    let original = |name| with_originals.then_some(name);
    vec![
      product(
        "sku-1",
        "blueberry gummies resin vape cartridge",
        original("blueberry gummies vegan fruit chews"),
      ),
      product(
        "sku-2",
        "sour diesel resin vape cartridge",
        original("sour diesel vape"),
      ),
      product("sku-3", "og kush pre-roll", None),
    ]
  }

  async fn backup_of<S: Scan>(
    store: &S,
    layout: &(String, usize),
    page_records: usize,
  ) -> (String, Exported, Vec<String>) {
    let mut skipped = Vec::new();
    let (bytes, exported) = export(
      store,
      &header(layout),
      Vec::new(),
      page_records,
      |id, reason| skipped.push(format!("{}: {reason}", id.unwrap_or("?"))),
      |_| {},
    )
    .await
    .unwrap();
    (String::from_utf8(bytes).unwrap(), exported, skipped)
  }

  async fn restoring(
    engine: &TestEngine,
    layout: &(String, usize),
    backup: &str,
    write: bool,
  ) -> anyhow::Result<(Summary, Vec<(String, Restored)>)> {
    let mut told = Vec::new();
    let summary = restore(
      engine,
      (&layout.0, layout.1),
      || Ok(backup.as_bytes()),
      write,
      |id, outcome| told.push((id.to_owned(), *outcome.as_ref().unwrap())),
      |_| {},
    )
    .await?;
    told.sort();
    Ok((summary, told))
  }

  fn outcomes(pairs: &[(&str, Restored)]) -> Vec<(String, Restored)> {
    pairs.iter().map(|(id, outcome)| ((*id).to_owned(), *outcome)).collect()
  }

  #[tokio::test]
  async fn a_backup_holds_every_record_of_this_layout_once_whatever_the_size_of_a_page() {
    let (engine, store, layout) = serving();
    engine.import(catalog(true)).await.unwrap();

    let (whole, exported, skipped) = backup_of(&store, &layout, 100).await;
    assert_eq!(
      exported,
      Exported {
        points: 3,
        records: 3,
        other_layout: 0,
        unreadable: 0
      }
    );
    assert_eq!(skipped, Vec::<String>::new());
    assert_eq!(file::check(whole.as_bytes()).unwrap().records, 3);
    for page_records in [1, 2, 3] {
      assert_eq!(
        backup_of(&store, &layout, page_records).await.0,
        whole,
        "pages of {page_records}"
      );
    }

    // What is kept of each record is its anchor as the store has it, and what the record was.
    let (_, records) = file::read(whole.as_bytes()).unwrap();
    for kept in records.map(Result::unwrap) {
      let stored = store.get(&kept.id).await.unwrap().unwrap();
      assert_eq!(kept, crate::engine::KeptAnchor::from(&stored));
    }
  }

  #[tokio::test]
  async fn a_lost_store_that_is_imported_again_and_restored_has_its_anchors_back() {
    let (before, lost, layout) = serving();
    before.import(catalog(true)).await.unwrap();
    let (backup, ..) = backup_of(&lost, &layout, 2).await;

    // The collection is gone. The catalog comes back from the caller's database, which knows what each product
    // is and not what it was created as.
    let (engine, store, layout) = serving();
    engine.import(catalog(false)).await.unwrap();
    let step = |name: &str| EvaluateInput {
      record: product("sku-1", name, None).record,
      mode: WriteMode::DryRun,
      threshold: None,
    };
    let further = "blueberry diesel resin vape cartridge";
    assert!(
      engine.evaluate(step(further)).await.unwrap().reasons.is_empty(),
      "measured from today"
    );

    // Asked first what it would do, it says, and does nothing.
    let would = outcomes(&[
      ("sku-1", Restored::Restored),
      ("sku-2", Restored::Restored),
      ("sku-3", Restored::Already),
    ]);
    let (summary, told) = restoring(&engine, &layout, &backup, false).await.unwrap();
    assert_eq!(told, would);
    assert_eq!((summary.records, summary.failed), (3, 0));
    let untouched = store.get("sku-1").await.unwrap().unwrap();
    assert_eq!(
      (untouched.revision, &untouched.anchor),
      (1, &untouched.vectors.composite)
    );

    let (summary, told) = restoring(&engine, &layout, &backup, true).await.unwrap();
    assert_eq!(told, would);
    assert_eq!(
      (
        summary.count(Restored::Restored),
        summary.count(Restored::Already),
        summary.count(Restored::Changed)
      ),
      (2, 1, 0)
    );
    for id in ["sku-1", "sku-2", "sku-3"] {
      let (was, is) = (
        lost.get(id).await.unwrap().unwrap(),
        store.get(id).await.unwrap().unwrap(),
      );
      assert_eq!(
        (&is.anchor, &is.anchor_quantities),
        (&was.anchor, &was.anchor_quantities),
        "{id}"
      );
    }
    // The same step is now measured from where the product began, and is too far.
    let reasons = engine.evaluate(step(further)).await.unwrap().reasons;
    assert!(
      reasons
        .iter()
        .any(|reason| matches!(reason, similarity_core::Reason::AnchorDrift { .. })),
      "{reasons:?}"
    );

    // Restoring again finds every anchor where it should be.
    let (summary, _) = restoring(&engine, &layout, &backup, true).await.unwrap();
    assert_eq!((summary.count(Restored::Already), summary.outcomes.len()), (3, 1));
  }

  #[tokio::test]
  async fn a_restore_says_of_each_record_it_left_alone_why() {
    let (before, lost, layout) = serving();
    before.import(catalog(true)).await.unwrap();
    let (backup, ..) = backup_of(&lost, &layout, 100).await;

    // Loaded again with one product edited since the backup was taken, one that came with the version it was
    // created as, and one that is not in the catalog any more.
    let (engine, _store, layout) = serving();
    engine
      .import(vec![
        product("sku-1", "blueberry gummies resin vape cartridge 1g", None),
        product("sku-2", "sour diesel resin vape cartridge", Some("sour diesel")),
      ])
      .await
      .unwrap();
    let (summary, told) = restoring(&engine, &layout, &backup, true).await.unwrap();
    assert_eq!(
      told,
      outcomes(&[
        ("sku-1", Restored::Changed),
        ("sku-2", Restored::HasHistory),
        ("sku-3", Restored::NotStored),
      ])
    );
    assert_eq!(
      (summary.records, summary.count(Restored::Restored), summary.failed),
      (3, 0, 0)
    );
  }

  #[tokio::test]
  async fn a_backup_of_another_layout_or_one_that_is_not_whole_is_refused_before_anything_is_written() {
    let (before, lost, layout) = serving();
    before.import(catalog(true)).await.unwrap();
    let (backup, ..) = backup_of(&lost, &layout, 100).await;
    let (engine, store, layout) = serving();
    engine.import(catalog(false)).await.unwrap();
    let refused = async |backup: &str, layout: &(String, usize)| {
      let error = format!("{:#}", restoring(&engine, layout, backup, true).await.unwrap_err());
      for id in ["sku-1", "sku-2", "sku-3"] {
        assert_eq!(store.get(id).await.unwrap().unwrap().revision, 1, "{id} was written");
      }
      error
    };

    // Another model, or the same model over other fields: the vectors are not this server's.
    let other = (format!("another-model{}", &layout.0), layout.1);
    let error = refused(&backup, &other).await;
    assert!(
      error.contains("the backup is of layout") && error.contains("another-model"),
      "{error}"
    );
    let wider = (layout.0.clone(), layout.1 + 64);
    assert!(refused(&backup, &wider).await.contains("components"));

    // Cut short after the second record: the first two could be restored, and are not.
    let cut: Vec<&str> = backup.lines().take(3).collect();
    assert!(refused(&cut.join("\n"), &layout).await.contains("cut short"));
    assert!(refused("", &layout).await.contains("empty"));
  }

  /// A store with something in it that is in no record's place, and whose second page cannot be read if it is
  /// told so.
  struct Mixed {
    store: Arc<MemoryStore>,
    fail_after_first_page: bool,
  }

  impl Scan for Mixed {
    async fn scan(&self, from: Option<&Cursor>, limit: usize, read: Read) -> similarity_store::Result<Page> {
      if from.is_some() && self.fail_after_first_page {
        return Err(StoreError::Unavailable("connection reset".to_owned()));
      }
      let mut page = self.store.scan(from, limit, read).await?;
      if page.next.is_none() {
        page.entries.push(Entry::Unreadable {
          id: None,
          reason: "it does not say which record it is".to_owned(),
        });
      }
      Ok(page)
    }
  }

  #[tokio::test]
  async fn what_cannot_be_kept_is_counted_and_named_and_a_store_that_fails_leaves_no_backup() {
    // A collection that outlived a change of model: one record under the old layout, two under this one.
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let layout = store_layout(&schema(), &embedder);
    let old = MemoryStore::new("another-model|category,name", layout.1);
    old
      .put(&similarity_store::StoredRecord {
        id: "old-1".to_owned(),
        brand_id: None,
        vectors: similarity_core::RecordVectors {
          composite: vec![1.0; layout.1],
          hashes: BTreeMap::new(),
        },
        anchor: vec![1.0; layout.1],
        quantities: similarity_core::Quantities::new(),
        anchor_quantities: similarity_core::Quantities::new(),
        revision: 1,
        updated_at: 1_700_000_000,
      })
      .await
      .unwrap();
    let store = Arc::new(old.reopened(layout.0.as_str(), layout.1));
    let engine = Engine::new(Arc::clone(&store), embedder, schema(), Policy::new(0.75, None).unwrap());
    engine
      .import(catalog(true).into_iter().take(2).collect())
      .await
      .unwrap();

    let whole = Mixed {
      store: Arc::clone(&store),
      fail_after_first_page: false,
    };
    let (backup, exported, mut skipped) = backup_of(&whole, &layout, 100).await;
    assert_eq!(
      exported,
      Exported {
        points: 4,
        records: 2,
        other_layout: 1,
        unreadable: 1
      }
    );
    skipped.sort();
    assert_eq!(
      skipped,
      [
        "?: it does not say which record it is",
        "old-1: it is stored under another layout, `another-model|category,name`",
      ]
    );
    assert_eq!(file::check(backup.as_bytes()).unwrap().records, 2);

    // A store that fails on its second page: an error, and what was written until then has no last line.
    let failing = Mixed {
      store,
      fail_after_first_page: true,
    };
    let mut written = Vec::new();
    let outcome = export(&failing, &header(&layout), &mut written, 2, |_, _| {}, |_| {}).await;
    let error = format!("{:#}", outcome.map(|(_, exported)| exported).unwrap_err());
    assert!(
      error.contains("first 2 point(s)") && error.contains("connection reset"),
      "{error}"
    );
    assert!(!written.is_empty(), "the first page was written");
    assert!(file::check(written.as_slice()).is_err(), "and is not a backup");
  }
}
