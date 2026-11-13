//! What an audit compares and what it counts: a stored record against what an export says it should be.
//!
//! Nothing here reads a file or a store. [`compare`] takes the two sides of one record and says where they
//! disagree, and [`Summary`] is the one place a count is kept, so that the report and the events cannot disagree
//! about a number (as `offline::stats` is for a calibration).
//!
//! What can be compared is what the store keeps besides vectors: the brand, the fingerprint of each field's
//! current text, and the current quantities. The anchor is a vector, with no fingerprints, and takes no part.

use std::collections::BTreeMap;

use similarity_core::quantity::{self, Quantities, QuantityChange};
use similarity_store::ScannedRecord;

/// What an export says a record should be stored as: what an import of that line would have written.
#[derive(Debug, Clone, PartialEq)]
pub struct Expected {
  /// The line of the export the record is on.
  pub line: usize,
  /// The key of the brand the export gives the record.
  pub brand_id: Option<String>,
  /// The fingerprint of each field's text (`PreparedRecord::hashes`).
  pub hashes: BTreeMap<String, String>,
  /// The record's quantities, validated.
  pub quantities: Quantities,
}

/// What kind of thing an audit found. The codes are an interface: add one, never rename or reuse one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
  /// A line of the export that an import would refuse. Its record was not compared.
  InvalidLine,
  /// A line whose id an earlier line already gave. An import skips it, and so does the audit.
  DuplicateId,
  /// The export has the record and the store does not.
  NotStored,
  /// The store has the record and the export does not.
  NotInExport,
  /// The record is stored under another brand than the export gives it, or the export gives it none. The server
  /// refuses every request for it that names the export's brand.
  BrandDiffers,
  /// The export gives the record a brand and it is stored without one. Such a record takes any request.
  BrandNotStored,
  /// A field's text is not the text that is stored.
  TextDiffers,
  /// A quantity is not the amount that is stored.
  QuantityDiffers,
  /// The record is stored under another layout: by another model or another set of fields. The server cannot
  /// read it.
  OtherLayout,
  /// Something in the store that cannot be read as a record.
  Unreadable,
}

impl Kind {
  /// Every kind, in the order a report lists them.
  pub const ALL: [Self; 10] = [
    Self::InvalidLine,
    Self::DuplicateId,
    Self::NotStored,
    Self::NotInExport,
    Self::BrandDiffers,
    Self::BrandNotStored,
    Self::TextDiffers,
    Self::QuantityDiffers,
    Self::OtherLayout,
    Self::Unreadable,
  ];

  /// The kind's name in an event.
  #[must_use]
  pub const fn code(self) -> &'static str {
    match self {
      Self::InvalidLine => "invalid_line",
      Self::DuplicateId => "duplicate_id",
      Self::NotStored => "not_stored",
      Self::NotInExport => "not_in_export",
      Self::BrandDiffers => "brand_differs",
      Self::BrandNotStored => "brand_not_stored",
      Self::TextDiffers => "text_differs",
      Self::QuantityDiffers => "quantity_differs",
      Self::OtherLayout => "other_layout",
      Self::Unreadable => "unreadable",
    }
  }
}

/// How a field or a quantity differs between the two sides. The codes are an interface, like [`Kind`]'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Difference {
  /// Both sides have a value, and they are not the same.
  Differs,
  /// The export has a value and the store has none.
  OnlyInExport,
  /// The store has a value and the export has none.
  OnlyInStore,
  /// Both sides have a quantity, measured in units that cannot be converted into one another.
  Incomparable,
}

impl Difference {
  /// The difference's name in an event.
  #[must_use]
  pub const fn code(self) -> &'static str {
    match self {
      Self::Differs => "differs",
      Self::OnlyInExport => "only_in_export",
      Self::OnlyInStore => "only_in_store",
      Self::Incomparable => "incomparable",
    }
  }
}

/// One thing an audit found. Flat, because it is written as one event (D17): every value that does not apply to
/// a finding's kind is `None`. It carries ids and brands, and never a record's text.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
  /// What was found.
  pub kind: Kind,
  /// The record's id, where it is known.
  pub id: Option<String>,
  /// The line of the export the record is on.
  pub line: Option<usize>,
  /// For a duplicate, the earlier line that has the same id.
  pub first_line: Option<usize>,
  /// The brand the export gives the record.
  pub brand_id: Option<String>,
  /// The brand the record is stored under.
  pub stored_brand_id: Option<String>,
  /// The text field that differs.
  pub field: Option<String>,
  /// The quantity that differs.
  pub quantity: Option<String>,
  /// How the field or the quantity differs.
  pub difference: Option<Difference>,
  /// How many times larger or smaller the export's amount is than the stored one, where there is a number to
  /// give (`QuantityScore::factor`).
  pub factor: Option<f64>,
  /// The export's unit for the quantity.
  pub unit: Option<String>,
  /// The stored unit for the quantity.
  pub stored_unit: Option<String>,
  /// The layout a record of another layout is stored under.
  pub layout: Option<String>,
  /// Why a line is invalid, or why a point cannot be read.
  pub reason: Option<String>,
  /// The stored record's revision.
  pub revision: Option<u64>,
  /// When the stored record was last written, in seconds since the Unix epoch.
  pub updated_at: Option<u64>,
  /// Whether the stored record was written at or after the time the export was taken. `None` when that time was
  /// not given, or when nothing is stored to have a time.
  pub after_export: Option<bool>,
}

impl Finding {
  /// A finding of `kind` about nothing yet: every other value is filled in by whoever found it.
  #[must_use]
  pub const fn new(kind: Kind) -> Self {
    Self {
      kind,
      id: None,
      line: None,
      first_line: None,
      brand_id: None,
      stored_brand_id: None,
      field: None,
      quantity: None,
      difference: None,
      factor: None,
      unit: None,
      stored_unit: None,
      layout: None,
      reason: None,
      revision: None,
      updated_at: None,
      after_export: None,
    }
  }

  /// A finding of `kind` about a stored record, with what the store says of it.
  #[must_use]
  pub fn stored(kind: Kind, stored: &ScannedRecord, exported_at: Option<u64>) -> Self {
    Self {
      id: Some(stored.id.clone()),
      stored_brand_id: stored.brand_id.clone(),
      revision: Some(stored.revision),
      updated_at: Some(stored.updated_at),
      after_export: written_since(stored.updated_at, exported_at),
      ..Self::new(kind)
    }
  }
}

/// Whether a record last written at `updated_at` was written since an export taken at `exported_at`, where that
/// is known. A record written in the very second the export was taken counts as written since: the export may
/// have read it before the write, and calling a difference that is not one costs more than setting one aside.
#[must_use]
pub fn written_since(updated_at: u64, exported_at: Option<u64>) -> Option<bool> {
  exported_at.map(|taken| updated_at >= taken)
}

/// Where a stored record is not what the export says it should be. Empty when the two agree.
///
/// Text is compared by fingerprint, so text that differs only in whitespace is the same text, as it is for the
/// server. Quantities are compared as the server compares them (`quantity::compare`): the same amount in another
/// unit is the same amount.
#[must_use]
pub fn compare(expected: &Expected, stored: &ScannedRecord, exported_at: Option<u64>) -> Vec<Finding> {
  let finding = |kind| Finding {
    line: Some(expected.line),
    brand_id: expected.brand_id.clone(),
    ..Finding::stored(kind, stored, exported_at)
  };
  let mut findings = Vec::new();

  match (&expected.brand_id, &stored.brand_id) {
    (given, kept) if given == kept => {}
    (Some(_), None) => findings.push(finding(Kind::BrandNotStored)),
    _ => findings.push(finding(Kind::BrandDiffers)),
  }

  let fields: std::collections::BTreeSet<&String> = expected.hashes.keys().chain(stored.hashes.keys()).collect();
  for field in fields {
    let difference = match (expected.hashes.get(field), stored.hashes.get(field)) {
      (Some(given), Some(kept)) if given == kept => continue,
      (Some(_), Some(_)) => Difference::Differs,
      (Some(_), None) => Difference::OnlyInExport,
      (None, _) => Difference::OnlyInStore,
    };
    findings.push(Finding {
      field: Some(field.clone()),
      difference: Some(difference),
      ..finding(Kind::TextDiffers)
    });
  }

  let quantities: std::collections::BTreeSet<&String> =
    expected.quantities.keys().chain(stored.quantities.keys()).collect();
  for name in quantities {
    // The stored amount is the one compared against, and the export's the one that is new to it: the way round
    // the server would see the export's record if it were sent as an edit.
    let Some(score) = quantity::compare(name, stored.quantities.get(name), None, expected.quantities.get(name)) else {
      continue;
    };
    let difference = match score.change {
      QuantityChange::Unchanged => continue,
      QuantityChange::Modified => Difference::Differs,
      QuantityChange::Incomparable => Difference::Incomparable,
      QuantityChange::Added | QuantityChange::Restored => Difference::OnlyInExport,
      QuantityChange::Removed => Difference::OnlyInStore,
    };
    findings.push(Finding {
      quantity: Some(name.clone()),
      difference: Some(difference),
      factor: score.factor,
      unit: score.unit,
      stored_unit: score.previous_unit,
      ..finding(Kind::QuantityDiffers)
    });
  }
  findings
}

/// What an audit counted. Every number a report or an event gives is one of these.
///
/// Everything an audit looks at ends as exactly one of [`Self::agree`], [`Self::differ`], [`Self::not_stored`],
/// [`Self::not_in_export`], and [`Self::not_compared`]: every record the export or the store has, and every line
/// and point that could not be told to be one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Summary {
  /// Lines of the export that held something.
  pub lines: usize,
  /// Lines an import would refuse.
  pub invalid_lines: usize,
  /// Lines whose id an earlier line had already given.
  pub duplicate_ids: usize,
  /// Records the export holds: [`Self::lines`] less the invalid and the duplicate ones.
  pub export_records: usize,
  /// Points the store holds, of any kind.
  pub points: usize,
  /// Records the store holds that this server can read.
  pub stored_records: usize,
  /// Records stored under another layout.
  pub other_layout: usize,
  /// How many records each of those layouts has.
  pub other_layouts: BTreeMap<String, usize>,
  /// Points that cannot be read as a record.
  pub unreadable: usize,
  /// Records on both sides that are the same.
  pub agree: usize,
  /// Records on both sides that are not.
  pub differ: usize,
  /// Records the export has and the store does not.
  pub not_stored: usize,
  /// Records the store has and the export does not.
  pub not_in_export: usize,
  /// What could not be compared: a record of another layout, a point that cannot be read, a record whose only
  /// lines in the export are invalid.
  pub not_compared: usize,
  /// Records stored under another brand than the export gives them.
  pub brand_differs: usize,
  /// Records the export gives a brand that are stored without one.
  pub brand_not_stored: usize,
  /// Records with at least one field whose text is not the stored text.
  pub text_differs: usize,
  /// Records with at least one quantity that is not the stored amount.
  pub quantity_differs: usize,
  /// Records that differ or are not in the export, and were written since the export was taken: an edit made
  /// since, as far as anyone can tell, and not a disagreement.
  pub after_export: usize,
  /// Findings reported.
  pub findings: usize,
}

impl Summary {
  /// Counts a record that is on both sides, by what [`compare`] found.
  pub fn compared(&mut self, findings: &[Finding]) {
    let Some(first) = findings.first() else {
      self.agree = self.agree.saturating_add(1);
      return;
    };
    self.differ = self.differ.saturating_add(1);
    // A record is counted once in each kind, however many of its fields or quantities differ.
    let has = |kind| usize::from(findings.iter().any(|finding| finding.kind == kind));
    self.brand_differs = self.brand_differs.saturating_add(has(Kind::BrandDiffers));
    self.brand_not_stored = self.brand_not_stored.saturating_add(has(Kind::BrandNotStored));
    self.text_differs = self.text_differs.saturating_add(has(Kind::TextDiffers));
    self.quantity_differs = self.quantity_differs.saturating_add(has(Kind::QuantityDiffers));
    self.after_export = self
      .after_export
      .saturating_add(usize::from(first.after_export == Some(true)));
  }

  /// How many things the export and the store disagree about, or could not be compared on: everything that does
  /// not agree, less what was written since the export was taken. Zero is a clean audit.
  #[must_use]
  pub const fn differences(&self) -> usize {
    self
      .differ
      .saturating_add(self.not_stored)
      .saturating_add(self.not_in_export)
      .saturating_add(self.not_compared)
      .saturating_sub(self.after_export)
  }
}

#[cfg(test)]
mod tests {
  use similarity_core::Quantity;

  use super::*;

  fn hashes(fields: &[(&str, &str)]) -> BTreeMap<String, String> {
    fields
      .iter()
      .map(|(name, text)| {
        let normalized = similarity_core::record::normalize_text(text).unwrap();
        ((*name).to_owned(), similarity_core::record::text_hash(&normalized))
      })
      .collect()
  }

  fn amounts(quantities: &[(&str, Quantity)]) -> Quantities {
    quantities
      .iter()
      .map(|(name, quantity)| ((*name).to_owned(), quantity.clone()))
      .collect()
  }

  fn expected(brand_id: Option<&str>, fields: &[(&str, &str)], quantities: &[(&str, Quantity)]) -> Expected {
    Expected {
      line: 7,
      brand_id: brand_id.map(str::to_owned),
      hashes: hashes(fields),
      quantities: amounts(quantities),
    }
  }

  fn stored(brand_id: Option<&str>, fields: &[(&str, &str)], quantities: &[(&str, Quantity)]) -> ScannedRecord {
    ScannedRecord {
      id: "sku-1".to_owned(),
      brand_id: brand_id.map(str::to_owned),
      hashes: hashes(fields),
      quantities: amounts(quantities),
      anchor_quantities: Quantities::new(),
      revision: 3,
      updated_at: 1_700_000_000,
      anchor: None,
    }
  }

  const GUMMIES: &[(&str, &str)] = &[("name", "blueberry gummies"), ("category", "edibles")];

  #[test]
  fn a_record_stored_as_the_export_says_has_no_findings() {
    let weight = [("weight", Quantity::single(40.0, "g"))];
    assert_eq!(
      compare(
        &expected(Some("brand-1"), GUMMIES, &weight),
        &stored(Some("brand-1"), GUMMIES, &weight),
        None
      ),
      []
    );
    // Neither side having a brand is agreeing about it.
    assert_eq!(
      compare(&expected(None, GUMMIES, &[]), &stored(None, GUMMIES, &[]), None),
      []
    );
  }

  #[test]
  fn text_that_differs_only_in_whitespace_is_not_a_difference() {
    let respaced = [("name", "  blueberry \n gummies"), ("category", "edibles\t")];
    assert_eq!(
      compare(&expected(None, &respaced, &[]), &stored(None, GUMMIES, &[]), None),
      []
    );
    // Case is a difference to the server, and so it is one here.
    let recased = [("name", "Blueberry gummies"), ("category", "edibles")];
    assert_eq!(
      compare(&expected(None, &recased, &[]), &stored(None, GUMMIES, &[]), None).len(),
      1
    );
  }

  #[test]
  fn a_changed_field_is_named_and_a_field_one_side_lacks_says_which_side() {
    let export = [("name", "peach gummies"), ("description", "soft and sweet")];
    let findings = compare(&expected(None, &export, &[]), &stored(None, GUMMIES, &[]), None);
    let told: Vec<(Kind, &str, Difference)> = findings
      .iter()
      .map(|finding| {
        (
          finding.kind,
          finding.field.as_deref().unwrap(),
          finding.difference.unwrap(),
        )
      })
      .collect();
    assert_eq!(
      told,
      [
        (Kind::TextDiffers, "category", Difference::OnlyInStore),
        (Kind::TextDiffers, "description", Difference::OnlyInExport),
        (Kind::TextDiffers, "name", Difference::Differs),
      ]
    );
    // Each says which record, where in the export, and which write of it is stored.
    for finding in &findings {
      assert_eq!(finding.id.as_deref(), Some("sku-1"));
      assert_eq!((finding.line, finding.revision), (Some(7), Some(3)));
      assert_eq!(finding.updated_at, Some(1_700_000_000));
      assert_eq!((finding.quantity.as_ref(), finding.factor), (None, None));
    }
  }

  #[test]
  fn a_quantity_in_another_unit_is_the_same_amount_and_another_amount_is_a_finding() {
    let kept = [
      ("weight", Quantity::single(1.0, "g")),
      ("volume", Quantity::single(30.0, "ml")),
      ("pieces", Quantity::single(10.0, "count")),
    ];
    let same = [
      ("weight", Quantity::single(1000.0, "mg")),
      ("volume", Quantity::single(30.0, " ML ")),
      ("pieces", Quantity::single(10.0, "count")),
    ];
    assert_eq!(
      compare(&expected(None, GUMMIES, &same), &stored(None, GUMMIES, &kept), None),
      []
    );

    let other = [
      ("weight", Quantity::single(3.5, "g")),
      ("volume", Quantity::single(30.0, "g")),
      ("thc", Quantity::single(10.0, "mg")),
    ];
    let findings = compare(&expected(None, GUMMIES, &other), &stored(None, GUMMIES, &kept), None);
    let told: Vec<String> = findings
      .iter()
      .map(|finding| {
        assert_eq!((finding.kind, finding.field.as_ref()), (Kind::QuantityDiffers, None));
        format!(
          "{} {} factor={:?} export={:?} stored={:?}",
          finding.quantity.as_deref().unwrap(),
          finding.difference.unwrap().code(),
          finding.factor,
          finding.unit.as_deref(),
          finding.stored_unit.as_deref(),
        )
      })
      .collect();
    assert_eq!(
      told,
      [
        r#"pieces only_in_store factor=None export=None stored=Some("count")"#,
        r#"thc only_in_export factor=None export=Some("mg") stored=None"#,
        r#"volume incomparable factor=None export=Some("g") stored=Some("ml")"#,
        r#"weight differs factor=Some(3.5) export=Some("g") stored=Some("g")"#,
      ]
    );
  }

  #[test]
  fn a_record_stored_without_a_brand_is_told_apart_from_one_stored_under_another() {
    let kinds = |given: Option<&str>, kept: Option<&str>| -> Vec<Kind> {
      compare(&expected(given, GUMMIES, &[]), &stored(kept, GUMMIES, &[]), None)
        .iter()
        .map(|finding| finding.kind)
        .collect()
    };
    assert_eq!(kinds(Some("brand-1"), Some("brand-1")), []);
    assert_eq!(kinds(Some("brand-1"), Some("brand-2")), [Kind::BrandDiffers]);
    // The server refuses a request that names no brand for a record that has one, so this is one too.
    assert_eq!(kinds(None, Some("brand-2")), [Kind::BrandDiffers]);
    assert_eq!(kinds(Some("brand-1"), None), [Kind::BrandNotStored]);

    let finding = compare(
      &expected(Some("brand-1"), GUMMIES, &[]),
      &stored(Some("brand-2"), GUMMIES, &[]),
      None,
    )
    .remove(0);
    assert_eq!(
      (finding.brand_id.as_deref(), finding.stored_brand_id.as_deref()),
      (Some("brand-1"), Some("brand-2"))
    );
  }

  #[test]
  fn a_finding_says_whether_the_record_was_written_since_the_export_only_when_that_time_is_known() {
    let since = |exported_at| {
      compare(
        &expected(Some("brand-1"), GUMMIES, &[]),
        &stored(Some("brand-2"), GUMMIES, &[]),
        exported_at,
      )
      .remove(0)
      .after_export
    };
    assert_eq!(since(None), None);
    assert_eq!(since(Some(1_700_000_001)), Some(false));
    assert_eq!(since(Some(1_699_999_999)), Some(true));
    assert_eq!(
      since(Some(1_700_000_000)),
      Some(true),
      "a record written in the second the export was taken counts as written since"
    );
  }

  #[test]
  fn a_record_with_several_findings_is_counted_once_in_each_kind() {
    let export = [("name", "peach gummies"), ("category", "candy")];
    let weights = (
      [("weight", Quantity::single(40.0, "g"))],
      [("weight", Quantity::single(80.0, "g"))],
    );
    let findings = compare(
      &expected(Some("brand-1"), &export, &weights.0),
      &stored(Some("brand-2"), GUMMIES, &weights.1),
      Some(1_700_000_500),
    );
    assert_eq!(findings.len(), 4, "a brand, two fields, and a quantity");

    let mut summary = Summary::default();
    summary.compared(&findings);
    summary.compared(&[]);
    summary.compared(&compare(
      &expected(Some("brand-1"), GUMMIES, &[]),
      &stored(None, GUMMIES, &[]),
      Some(1_600_000_000),
    ));
    assert_eq!(
      summary,
      Summary {
        agree: 1,
        differ: 2,
        brand_differs: 1,
        brand_not_stored: 1,
        text_differs: 1,
        quantity_differs: 1,
        after_export: 1,
        ..Summary::default()
      }
    );
  }

  #[test]
  fn what_was_written_since_the_export_is_not_a_difference_and_everything_else_that_does_not_agree_is() {
    let summary = Summary {
      agree: 100,
      differ: 5,
      not_stored: 3,
      not_in_export: 2,
      not_compared: 1,
      after_export: 4,
      ..Summary::default()
    };
    assert_eq!(summary.differences(), 7);
    assert_eq!(
      Summary {
        agree: 100,
        ..Summary::default()
      }
      .differences(),
      0
    );
    for one in [
      Summary {
        differ: 1,
        ..Summary::default()
      },
      Summary {
        not_stored: 1,
        ..Summary::default()
      },
      Summary {
        not_in_export: 1,
        ..Summary::default()
      },
      Summary {
        not_compared: 1,
        ..Summary::default()
      },
    ] {
      assert_eq!(one.differences(), 1, "{one:?}");
    }
  }

  #[test]
  fn every_kind_and_every_difference_has_its_own_code() {
    let codes: std::collections::BTreeSet<&str> = Kind::ALL.iter().map(|kind| kind.code()).collect();
    assert_eq!(codes.len(), Kind::ALL.len());
    // These are read by other tools. A code is added, never changed.
    assert_eq!(
      Kind::ALL.map(Kind::code),
      [
        "invalid_line",
        "duplicate_id",
        "not_stored",
        "not_in_export",
        "brand_differs",
        "brand_not_stored",
        "text_differs",
        "quantity_differs",
        "other_layout",
        "unreadable"
      ]
    );
    assert_eq!(
      [
        Difference::Differs,
        Difference::OnlyInExport,
        Difference::OnlyInStore,
        Difference::Incomparable
      ]
      .map(Difference::code),
      ["differs", "only_in_export", "only_in_store", "incomparable"]
    );
  }
}
