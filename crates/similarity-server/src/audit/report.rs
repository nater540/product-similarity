//! An audit for a person, and the same audit as data.
//!
//! The report is text on stdout: the counts, and the first few findings of each kind. The events are one flat
//! line for each finding and one for the summary (D17). Every count in either comes from the audit's
//! [`Summary`], so the two cannot disagree, and neither holds a record's text: a line carries ids and brands.
//!
//! The event names and their fields are an interface once written, like a calibration's: a field is added, never
//! renamed or given another meaning. `audit_summary` is the last line of an audit that reached its end. An
//! output without one is of an audit that did not, and its findings are not all of them.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use similarity_log::RESULT;

use super::{Finding, Kind, Summary};

/// What was audited against what.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Audited<'a> {
  /// The export, as it was named on the command line.
  pub export: &'a str,
  /// The collection the store keeps its records in.
  pub collection: &'a str,
  /// The layout this server's records are stored under.
  pub layout: &'a str,
  /// When the export was taken, where that was given.
  pub exported_at: Option<u64>,
}

/// The findings a report shows: the first few of each kind, and how many of each there were.
#[derive(Debug, Clone, PartialEq)]
pub struct Shown {
  limit: usize,
  kept: BTreeMap<Kind, Vec<Finding>>,
  seen: BTreeMap<Kind, usize>,
}

impl Shown {
  /// Keeps at most `limit` findings of each kind.
  #[must_use]
  pub const fn new(limit: usize) -> Self {
    Self {
      limit,
      kept: BTreeMap::new(),
      seen: BTreeMap::new(),
    }
  }

  /// Takes a finding as it is made.
  pub fn take(&mut self, finding: &Finding) {
    let seen = self.seen.entry(finding.kind).or_default();
    *seen = seen.saturating_add(1);
    let kept = self.kept.entry(finding.kind).or_default();
    if kept.len() < self.limit {
      kept.push(finding.clone());
    }
  }
}

/// Writes one finding as an `audit_finding` event. What does not apply to its kind is left out.
pub fn emit_finding(finding: &Finding) {
  tracing::info!(
    target: RESULT,
    kind = finding.kind.code(),
    id = finding.id.as_deref(),
    line = finding.line,
    first_line = finding.first_line,
    brand_id = finding.brand_id.as_deref(),
    stored_brand_id = finding.stored_brand_id.as_deref(),
    field = finding.field.as_deref(),
    quantity = finding.quantity.as_deref(),
    difference = finding.difference.map(super::Difference::code),
    factor = finding.factor,
    unit = finding.unit.as_deref(),
    stored_unit = finding.stored_unit.as_deref(),
    layout = finding.layout.as_deref(),
    reason = finding.reason.as_deref(),
    revision = finding.revision,
    updated_at = finding.updated_at,
    after_export = finding.after_export,
    "audit_finding"
  );
}

/// Writes what an audit counted: an `audit_layout` event for each other layout the store holds records of, and
/// then `audit_summary`, the last line of an audit that reached its end.
pub fn emit_summary(audited: Audited<'_>, summary: &Summary, elapsed_s: f64) {
  for (layout, records) in &summary.other_layouts {
    tracing::info!(target: RESULT, layout = layout.as_str(), records, "audit_layout");
  }
  tracing::info!(
    target: RESULT,
    export = audited.export,
    collection = audited.collection,
    layout = audited.layout,
    exported_at = audited.exported_at,
    lines = summary.lines,
    invalid_lines = summary.invalid_lines,
    duplicate_ids = summary.duplicate_ids,
    export_records = summary.export_records,
    points = summary.points,
    stored_records = summary.stored_records,
    other_layout = summary.other_layout,
    unreadable = summary.unreadable,
    agree = summary.agree,
    differ = summary.differ,
    not_stored = summary.not_stored,
    not_in_export = summary.not_in_export,
    not_compared = summary.not_compared,
    brand_differs = summary.brand_differs,
    brand_not_stored = summary.brand_not_stored,
    text_differs = summary.text_differs,
    quantity_differs = summary.quantity_differs,
    after_export = summary.after_export,
    findings = summary.findings,
    differences = summary.differences(),
    elapsed_s,
    "audit_summary"
  );
}

/// What a kind of finding is called at the head of its list.
const fn heading(kind: Kind) -> &'static str {
  match kind {
    Kind::InvalidLine => "lines of the export an import would refuse",
    Kind::DuplicateId => "lines whose id an earlier line has (an import skips them)",
    Kind::NotStored => "in the export and not stored",
    Kind::NotInExport => "stored and not in the export",
    Kind::BrandDiffers => "stored under another brand",
    Kind::BrandNotStored => "stored without the brand the export gives",
    Kind::TextDiffers => "text that is not the stored text",
    Kind::QuantityDiffers => "quantities that are not the stored amount",
    Kind::OtherLayout => "stored under another layout (this server cannot read them)",
    Kind::Unreadable => "in the store and not readable as a record",
  }
}

/// One finding as a line of the report.
fn line(finding: &Finding) -> String {
  let mut text = String::from(" ");
  let mut part = |part: String| {
    text.push(' ');
    text.push_str(&part);
  };
  if let Some(line) = finding.line {
    part(format!("line {line}"));
  }
  if let Some(id) = &finding.id {
    part(format!("`{id}`"));
  }
  if let Some(first) = finding.first_line {
    part(format!("first on line {first}"));
  }
  if let Some(name) = finding.field.as_ref().or(finding.quantity.as_ref()) {
    part(format!("`{name}`"));
  }
  if let Some(difference) = finding.difference {
    part(
      match difference {
        super::Difference::Differs => "differs",
        super::Difference::OnlyInExport => "only in the export",
        super::Difference::OnlyInStore => "only in the store",
        super::Difference::Incomparable => "in units that cannot be compared",
      }
      .to_owned(),
    );
  }
  if let Some(factor) = finding.factor {
    part(format!("by a factor of {factor:.2}"));
  }
  if let (Some(unit), Some(stored)) = (&finding.unit, &finding.stored_unit)
    && unit != stored
  {
    part(format!("(`{unit}` in the export, `{stored}` stored)"));
  }
  match finding.kind {
    Kind::BrandDiffers | Kind::BrandNotStored => {
      let name = |brand: &Option<String>| {
        brand
          .as_ref()
          .map_or_else(|| "none".to_owned(), |brand| format!("`{brand}`"))
      };
      part(format!(
        "export {}, stored {}",
        name(&finding.brand_id),
        name(&finding.stored_brand_id)
      ));
    }
    Kind::NotInExport => {
      if let Some(brand) = &finding.stored_brand_id {
        part(format!("brand `{brand}`"));
      }
    }
    Kind::NotStored => {
      if let Some(brand) = &finding.brand_id {
        part(format!("brand `{brand}`"));
      }
    }
    _ => {}
  }
  if let Some(layout) = &finding.layout {
    part(format!("layout `{layout}`"));
  }
  if let Some(reason) = &finding.reason {
    part(reason.clone());
  }
  if let (Some(revision), Some(updated_at)) = (finding.revision, finding.updated_at) {
    part(format!("(revision {revision}, written {updated_at})"));
  }
  if finding.after_export == Some(true) {
    part("written since the export".to_owned());
  }
  text
}

/// The report of an audit that reached its end.
#[must_use]
pub fn render(audited: Audited<'_>, summary: &Summary, shown: &Shown) -> String {
  let mut out = String::new();
  // Writing to a `String` cannot fail.
  let _ = writeln!(out, "export      {}", audited.export);
  let _ = writeln!(out, "collection  {}", audited.collection);
  let _ = writeln!(out, "layout      {}", audited.layout);
  if let Some(taken) = audited.exported_at {
    let _ = writeln!(out, "exported at {taken}");
  }
  let _ = writeln!(
    out,
    "\nexport      {} lines: {} records, {} invalid, {} with an id an earlier line has",
    summary.lines, summary.export_records, summary.invalid_lines, summary.duplicate_ids
  );
  let _ = writeln!(
    out,
    "store       {} points: {} records of this layout, {} of another, {} unreadable",
    summary.points, summary.stored_records, summary.other_layout, summary.unreadable
  );
  for (layout, records) in &summary.other_layouts {
    let _ = writeln!(out, "            {records} under `{layout}`");
  }

  let _ = writeln!(out, "\n{:>9}  agree", summary.agree);
  let _ = writeln!(
    out,
    "{:>9}  differ: {} in brand, {} stored without a brand, {} in text, {} in quantities",
    summary.differ, summary.brand_differs, summary.brand_not_stored, summary.text_differs, summary.quantity_differs
  );
  let _ = writeln!(out, "{:>9}  in the export and not stored", summary.not_stored);
  let _ = writeln!(out, "{:>9}  stored and not in the export", summary.not_in_export);
  let _ = writeln!(out, "{:>9}  could not be compared", summary.not_compared);
  if audited.exported_at.is_some() {
    let _ = writeln!(
      out,
      "{:>9}  of those were written since the export was taken, and are not counted below",
      summary.after_export
    );
  }
  let _ = writeln!(out, "{:>9}  differences", summary.differences());

  for kind in Kind::ALL {
    let Some(kept) = shown.kept.get(&kind) else {
      continue;
    };
    let seen = shown.seen.get(&kind).copied().unwrap_or_default();
    // A record is one finding for each field or quantity that differs, and is counted once.
    let count = match kind {
      Kind::TextDiffers => format!("{seen} fields of {} records", summary.text_differs),
      Kind::QuantityDiffers => format!("{seen} quantities of {} records", summary.quantity_differs),
      _ => seen.to_string(),
    };
    let _ = writeln!(out, "\n{} ({count}, `{}`)", heading(kind), kind.code());
    for finding in kept {
      let _ = writeln!(out, "{}", line(finding));
    }
    let left_out = seen.saturating_sub(kept.len());
    if left_out > 0 {
      let _ = writeln!(
        out,
        "  and {left_out} more: `--ndjson` writes every one, `--limit` shows more here"
      );
    }
  }

  let _ = writeln!(
    out,
    "\nA record's anchor is not compared: the store keeps a vector of it and no fingerprint of its text.\nThe store \
     was listed while it could be written to. Audit again before acting on a record that is missing on one side."
  );
  out
}

#[cfg(test)]
mod tests {
  use serde_json::Value;
  use similarity_core::Quantity;
  use similarity_store::ScannedRecord;

  use super::super::{Difference, Expected, compare};
  use super::*;

  const AUDITED: Audited<'_> = Audited {
    export: "export.jsonl",
    collection: "records",
    layout: "model-a-d64-t128|category,name",
    exported_at: Some(1_700_000_500),
  };

  /// An audit's worth of findings of several kinds, made the way an audit makes them, and its counts.
  fn audit() -> (Summary, Vec<Finding>) {
    let fingerprint = |text: &str| similarity_core::record::text_hash(text);
    let expected = Expected {
      line: 3,
      brand_id: Some("brand-1".to_owned()),
      hashes: [("name".to_owned(), fingerprint("peach gummies"))].into(),
      quantities: [("weight".to_owned(), Quantity::single(80.0, "g"))].into(),
    };
    let stored = ScannedRecord {
      id: "sku-3".to_owned(),
      brand_id: Some("brand-2".to_owned()),
      hashes: [("name".to_owned(), fingerprint("blueberry gummies"))].into(),
      quantities: [("weight".to_owned(), Quantity::single(40_000.0, "mg"))].into(),
      anchor_quantities: similarity_core::Quantities::new(),
      revision: 4,
      updated_at: 1_700_000_900,
      anchor: None,
    };
    let mut findings = compare(&expected, &stored, AUDITED.exported_at);
    let mut summary = Summary::default();
    summary.compared(&findings);
    for line in 10..17 {
      findings.push(Finding {
        id: Some(format!("sku-{line}")),
        line: Some(line),
        brand_id: Some("brand-1".to_owned()),
        ..Finding::new(Kind::NotStored)
      });
    }
    findings.push(Finding {
      line: Some(20),
      reason: Some("not a record to import: expected value at line 1 column 1".to_owned()),
      ..Finding::new(Kind::InvalidLine)
    });
    findings.push(Finding {
      id: Some("old-1".to_owned()),
      layout: Some("model-b-d64-t128|category,name".to_owned()),
      ..Finding::new(Kind::OtherLayout)
    });
    let summary = Summary {
      lines: 21,
      invalid_lines: 1,
      export_records: 20,
      points: 14,
      stored_records: 13,
      other_layout: 1,
      other_layouts: [("model-b-d64-t128|category,name".to_owned(), 1)].into(),
      agree: 12,
      not_stored: 7,
      not_compared: 2,
      findings: findings.len(),
      ..summary
    };
    (summary, findings)
  }

  fn events(summary: &Summary, findings: &[Finding]) -> Vec<Value> {
    let captured = similarity_log::testing::capture("off", || {
      for finding in findings {
        emit_finding(finding);
      }
      emit_summary(AUDITED, summary, 1.5);
    });
    captured
      .results
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect()
  }

  #[test]
  fn every_finding_is_one_flat_line_and_the_summary_comes_last() {
    let (summary, findings) = audit();
    let events = events(&summary, &findings);
    for event in &events {
      for (key, value) in event.as_object().unwrap() {
        assert!(
          !value.is_object() && !value.is_array() && !value.is_null(),
          "`{key}` in {event}"
        );
      }
    }
    let messages: Vec<&str> = events.iter().map(|event| event["message"].as_str().unwrap()).collect();
    assert_eq!(messages.len(), findings.len() + 2);
    assert!(
      messages[..findings.len()]
        .iter()
        .all(|message| *message == "audit_finding")
    );
    assert_eq!(messages[findings.len()..], ["audit_layout", "audit_summary"]);

    // A finding carries what applies to its kind, by these names, and nothing that does not.
    let brand = &events[0];
    assert_eq!(
      (
        brand["kind"].as_str(),
        brand["id"].as_str(),
        brand["line"].as_u64(),
        brand["brand_id"].as_str(),
        brand["stored_brand_id"].as_str(),
        brand["revision"].as_u64(),
        brand["updated_at"].as_u64(),
        brand["after_export"].as_bool(),
      ),
      (
        Some("brand_differs"),
        Some("sku-3"),
        Some(3),
        Some("brand-1"),
        Some("brand-2"),
        Some(4),
        Some(1_700_000_900),
        Some(true)
      )
    );
    for absent in [
      "field",
      "quantity",
      "difference",
      "factor",
      "layout",
      "reason",
      "first_line",
    ] {
      assert!(brand.get(absent).is_none(), "`{absent}` in {brand}");
    }
    let text = &events[1];
    assert_eq!(
      (
        text["kind"].as_str(),
        text["field"].as_str(),
        text["difference"].as_str()
      ),
      (Some("text_differs"), Some("name"), Some(Difference::Differs.code()))
    );
    let quantity = &events[2];
    assert_eq!(
      (
        quantity["kind"].as_str(),
        quantity["quantity"].as_str(),
        quantity["difference"].as_str(),
        quantity["factor"].as_f64(),
        quantity["unit"].as_str(),
        quantity["stored_unit"].as_str(),
      ),
      (
        Some("quantity_differs"),
        Some("weight"),
        Some("differs"),
        Some(2.0),
        Some("g"),
        Some("mg")
      )
    );
    let layout = &events[findings.len()];
    assert_eq!(
      (layout["layout"].as_str(), layout["records"].as_u64()),
      (Some("model-b-d64-t128|category,name"), Some(1))
    );
  }

  #[test]
  fn the_report_and_the_events_show_the_same_counts() {
    let (summary, findings) = audit();
    let events = events(&summary, &findings);
    let last = events.last().unwrap();
    let mut shown = Shown::new(3);
    for finding in &findings {
      shown.take(finding);
    }
    let text = render(AUDITED, &summary, &shown);

    let counts = [
      ("lines", summary.lines, "export      21 lines"),
      ("export_records", summary.export_records, " 20 records,"),
      ("invalid_lines", summary.invalid_lines, " 1 invalid,"),
      (
        "duplicate_ids",
        summary.duplicate_ids,
        " 0 with an id an earlier line has",
      ),
      ("points", summary.points, "store       14 points"),
      ("stored_records", summary.stored_records, " 13 records of this layout,"),
      ("other_layout", summary.other_layout, " 1 of another,"),
      ("unreadable", summary.unreadable, " 0 unreadable"),
      ("agree", summary.agree, "       12  agree"),
      ("differ", summary.differ, "        1  differ:"),
      ("brand_differs", summary.brand_differs, " 1 in brand,"),
      (
        "brand_not_stored",
        summary.brand_not_stored,
        " 0 stored without a brand,",
      ),
      ("text_differs", summary.text_differs, " 1 in text,"),
      ("quantity_differs", summary.quantity_differs, " 1 in quantities"),
      (
        "not_stored",
        summary.not_stored,
        "        7  in the export and not stored",
      ),
      (
        "not_in_export",
        summary.not_in_export,
        "        0  stored and not in the export",
      ),
      ("not_compared", summary.not_compared, "        2  could not be compared"),
      (
        "after_export",
        summary.after_export,
        "        1  of those were written since",
      ),
      ("differences", summary.differences(), "        9  differences"),
    ];
    for (name, count, in_the_report) in counts {
      assert_eq!(last[name].as_u64(), Some(count as u64), "`{name}` in {last}");
      assert!(text.contains(in_the_report), "{in_the_report:?} is not in:\n{text}");
    }
    assert_eq!(last["findings"].as_u64(), Some(findings.len() as u64));
    assert_eq!(
      (
        last["export"].as_str(),
        last["collection"].as_str(),
        last["layout"].as_str(),
        last["exported_at"].as_u64(),
        last["elapsed_s"].as_f64(),
      ),
      (
        Some("export.jsonl"),
        Some("records"),
        Some("model-a-d64-t128|category,name"),
        Some(1_700_000_500),
        Some(1.5)
      )
    );
    assert!(text.contains("exported at 1700000500\n"), "{text}");
    assert!(text.contains("1 under `model-b-d64-t128|category,name`"), "{text}");
  }

  #[test]
  fn the_report_lists_no_more_than_the_limit_of_each_kind_and_says_how_many_it_left_out() {
    let (summary, findings) = audit();
    let mut shown = Shown::new(3);
    for finding in &findings {
      shown.take(finding);
    }
    let text = render(AUDITED, &summary, &shown);

    assert!(
      text.contains(
        "\nin the export and not stored (7, `not_stored`)\n  line 10 `sku-10` brand `brand-1`\n  line 11 `sku-11` \
         brand `brand-1`\n  line 12 `sku-12` brand `brand-1`\n  and 4 more: "
      ),
      "{text}"
    );
    assert!(!text.contains("sku-13"), "{text}");
    // A kind with no more than the limit is listed whole and says nothing of more.
    assert!(
      text.contains(
        "\nstored under another brand (1, `brand_differs`)\n  line 3 `sku-3` export `brand-1`, stored `brand-2` \
         (revision 4, written 1700000900) written since the export\n\n"
      ),
      "{text}"
    );
    assert!(
      text.contains(
        "\nquantities that are not the stored amount (1 quantities of 1 records, `quantity_differs`)\n  line 3 `sku-3` `weight` differs by \
         a factor of 2.00 (`g` in the export, `mg` stored) (revision 4, written 1700000900) written since the \
         export\n"
      ),
      "{text}"
    );
    assert!(
      text.contains("\nlines of the export an import would refuse (1, `invalid_line`)\n  line 20 not a record"),
      "{text}"
    );
    assert!(
      text.contains("(1, `other_layout`)\n  `old-1` layout `model-b-d64-t128|category,name`\n"),
      "{text}"
    );
    // Kinds of which nothing was found have no list, and what an audit cannot see is said every time.
    assert!(
      !text.contains("`not_in_export`") && !text.contains("`unreadable`"),
      "{text}"
    );
    assert!(text.contains("A record's anchor is not compared"), "{text}");

    // With no time given for the export, nothing is said about what was written since.
    let untimed = Audited {
      exported_at: None,
      ..AUDITED
    };
    let text = render(untimed, &summary, &Shown::new(0));
    assert!(
      !text.contains("exported at") && !text.contains("written since"),
      "{text}"
    );
    assert!(!text.contains("sku-"), "a limit of nothing lists nothing: {text}");
  }

  #[test]
  fn no_line_carries_a_records_text() {
    let (summary, findings) = audit();
    let events = events(&summary, &findings);
    let mut shown = Shown::new(100);
    for finding in &findings {
      shown.take(finding);
    }
    let text = render(AUDITED, &summary, &shown);
    let written: String = events.iter().map(Value::to_string).collect::<String>() + &text;
    for word in ["peach", "blueberry", "gummies"] {
      assert!(!written.contains(word), "`{word}` was written");
    }
    // Nor its fingerprint, which is no secret and of no use to a reader.
    assert!(!written.contains(&similarity_core::record::text_hash("peach gummies")));
  }
}
