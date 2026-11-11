//! Comparing two calibrations: did a new model, new thresholds, new weights, or new code make verdicts worse?
//!
//! Both sides are runs read back from `calibrate --ndjson` ([`super::events::read`]). The pairs are matched by id,
//! and what is reported is what changed: every verdict that flipped, the totals, the slices whose errors moved,
//! and the fields whose ranges moved.
//!
//! The second run does not have to exist. Weights, floors, thresholds, and quantity limits are all applied to
//! measurements the first run already wrote down (the per-field similarities and per-quantity factors), so
//! [`rejudge`] produces "the same pairs under another configuration" with no model loaded and nothing embedded.
//! That covers most experiments, since none of those settings needs a re-index either (D12). A change of model or
//! of the set of fields changes the measurements themselves, and needs a real second run.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use anyhow::bail;
use similarity_core::{Policy, Schema};
use similarity_log::RESULT;

use super::calibrate::{Calibration, ScoredPair, summarize};
use super::events::{Run, six};
use super::pairs::{Input, Label};
use super::stats::{self, Range};

/// Two bounds closer than this are the same bound: it is half of the last digit the reports print.
const MOVED: f64 = 0.0005;

/// What a changed verdict means, given the pair's label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flip {
  /// An edit that was accepted is now rejected.
  NewFalseReject,
  /// A recycle that was rejected is now accepted.
  NewFalseAccept,
  /// A pair the old run got wrong is now right.
  Fixed,
  /// The verdict on a pair with any other label changed. Neither better nor worse; worth a look.
  Unlabeled,
}

impl Flip {
  /// The flip as the events spell it.
  pub fn as_str(self) -> &'static str {
    match self {
      Self::NewFalseReject => "new_false_reject",
      Self::NewFalseAccept => "new_false_accept",
      Self::Fixed => "fixed",
      Self::Unlabeled => "unlabeled",
    }
  }

  fn headline(self) -> &'static str {
    match self {
      Self::NewFalseReject => "NEW FALSE REJECT",
      Self::NewFalseAccept => "NEW FALSE ACCEPT",
      Self::Fixed => "fixed",
      Self::Unlabeled => "changed",
    }
  }
}

/// One pair whose verdict differs between the two runs.
#[derive(Debug, Clone, PartialEq)]
pub struct Flipped {
  /// The pair's id.
  pub id: String,
  /// Its label, as written.
  pub label: String,
  /// Its note.
  pub note: String,
  /// What the change means.
  pub flip: Flip,
  /// Its overall similarity in the old run and in the new.
  pub similarity: (f32, f32),
  /// Whether it was accepted in the old run and in the new.
  pub accepted: (bool, bool),
}

/// What a run gets wrong, and where its threshold would best sit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Totals {
  /// Edits rejected.
  pub false_rejects: usize,
  /// Recycles accepted.
  pub false_accepts: usize,
  /// The best overall threshold for the pairs, and its margin, if both labels are present.
  pub suggestion: Option<(f32, f32)>,
}

impl Totals {
  fn of(calibration: &Calibration) -> Self {
    Self {
      false_rejects: calibration.false_rejects,
      false_accepts: calibration.false_accepts,
      suggestion: calibration
        .suggestion
        .map(|suggestion| (suggestion.threshold, suggestion.margin)),
    }
  }

  fn errors(&self) -> usize {
    self.false_rejects.saturating_add(self.false_accepts)
  }
}

/// A slice whose error counts differ between the two runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceChange {
  /// The attribute the pairs are sliced by.
  pub by: &'static str,
  /// The value the slice's pairs share.
  pub value: String,
  /// Edits rejected, in the old run and in the new.
  pub false_rejects: (usize, usize),
  /// Recycles accepted, in the old run and in the new.
  pub false_accepts: (usize, usize),
}

/// A field whose range of similarity, among the pairs of one label that changed it, moved between the runs.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeShift {
  /// The field.
  pub field: String,
  /// The label.
  pub label: Label,
  /// The range in the old run.
  pub old: Option<Range>,
  /// The range in the new run.
  pub new: Option<Range>,
}

/// Everything that differs between two runs over the pairs they share.
#[derive(Debug, Clone, PartialEq)]
pub struct Diff {
  /// How many pairs both runs have.
  pub compared: usize,
  /// Ids of pairs only the old run has.
  pub only_old: Vec<String>,
  /// Ids of pairs only the new run has.
  pub only_new: Vec<String>,
  /// Every pair whose verdict changed, the largest change in score first.
  pub flips: Vec<Flipped>,
  /// The old run's totals, over the shared pairs.
  pub old: Totals,
  /// The new run's totals, over the shared pairs.
  pub new: Totals,
  /// Slices whose error counts changed.
  pub slices: Vec<SliceChange>,
  /// Fields whose ranges moved.
  pub fields: Vec<RangeShift>,
}

/// Whether two runs scored the same selection of the same file.
pub fn same_pairs(old: &Input, new: &Input) -> bool {
  (old.sha256.as_str(), &old.split, old.min_confidence) == (new.sha256.as_str(), &new.split, new.min_confidence)
}

fn restricted(calibration: &Calibration, ids: &BTreeSet<&str>) -> Calibration {
  let pairs = calibration
    .pairs
    .iter()
    .filter(|scored| ids.contains(scored.pair.id.as_str()))
    .cloned()
    .collect();
  summarize(pairs, calibration.input.clone())
}

/// What it means that `new`'s verdict differs from the one the pair had before.
fn flip_of(new: &ScoredPair) -> Flip {
  match (new.pair.label, new.evaluation.accepted()) {
    (None, _) => Flip::Unlabeled,
    (Some(Label::Edit), false) => Flip::NewFalseReject,
    (Some(Label::Recycle), true) => Flip::NewFalseAccept,
    // The new verdict is right, so the old one, being different, was wrong.
    (Some(_), _) => Flip::Fixed,
  }
}

fn moved(old: Option<Range>, new: Option<Range>) -> bool {
  match (old, new) {
    (None, None) => false,
    (Some(old), Some(new)) => {
      old.count != new.count || (old.low - new.low).abs() > MOVED || (old.high - new.high).abs() > MOVED
    }
    _ => true,
  }
}

/// Compares two calibrations over the pairs both have.
///
/// The new run's label is the one used for a pair, should the two disagree; the pairs are matched by id alone.
pub fn diff(old: &Calibration, new: &Calibration) -> Diff {
  let ids = |calibration: &Calibration| -> BTreeSet<String> {
    calibration.pairs.iter().map(|scored| scored.pair.id.clone()).collect()
  };
  let (old_ids, new_ids) = (ids(old), ids(new));
  let shared: BTreeSet<&str> = old_ids.intersection(&new_ids).map(String::as_str).collect();
  let (old, new) = (restricted(old, &shared), restricted(new, &shared));

  let by_id: BTreeMap<&str, &ScoredPair> = old
    .pairs
    .iter()
    .map(|scored| (scored.pair.id.as_str(), scored))
    .collect();
  let mut flips: Vec<Flipped> = new
    .pairs
    .iter()
    .filter_map(|after| {
      let before = by_id.get(after.pair.id.as_str())?;
      (before.evaluation.accepted() != after.evaluation.accepted()).then(|| Flipped {
        id: after.pair.id.clone(),
        label: after.pair.label_text.clone(),
        note: after.pair.note.clone(),
        flip: flip_of(after),
        similarity: (before.similarity(), after.similarity()),
        accepted: (before.evaluation.accepted(), after.evaluation.accepted()),
      })
    })
    .collect();
  let change = |flipped: &Flipped| (flipped.similarity.1 - flipped.similarity.0).abs();
  flips.sort_by(|a, b| change(b).total_cmp(&change(a)).then_with(|| a.id.cmp(&b.id)));

  let old_slices = stats::slices(&old);
  let slices = stats::slices(&new)
    .into_iter()
    .filter_map(|after| {
      let before = old_slices
        .iter()
        .find(|before| (before.by, &before.value) == (after.by, &after.value))?;
      let change = SliceChange {
        by: after.by,
        value: after.value,
        false_rejects: (before.false_rejects, after.false_rejects),
        false_accepts: (before.false_accepts, after.false_accepts),
      };
      (change.false_rejects.0 != change.false_rejects.1 || change.false_accepts.0 != change.false_accepts.1)
        .then_some(change)
    })
    .collect();

  let names: BTreeSet<&str> = stats::field_names(&old)
    .union(&stats::field_names(&new))
    .copied()
    .collect();
  let mut fields = Vec::new();
  for field in names {
    for label in [Label::Edit, Label::Recycle] {
      let (before, after) = (
        stats::field_range(&old, field, label),
        stats::field_range(&new, field, label),
      );
      if moved(before, after) {
        fields.push(RangeShift {
          field: field.to_owned(),
          label,
          old: before,
          new: after,
        });
      }
    }
  }

  Diff {
    compared: shared.len(),
    only_old: old_ids.difference(&new_ids).cloned().collect(),
    only_new: new_ids.difference(&old_ids).cloned().collect(),
    flips,
    old: Totals::of(&old),
    new: Totals::of(&new),
    slices,
    fields,
  }
}

/// The pairs of `run`, judged again under `schema` and `policy`, with no model loaded.
///
/// # Errors
///
/// Returns an error if the schema's text fields or quantities are not the ones the run was scored with. Their
/// weights and limits may differ, which is the point; their names may not, because a field the run never
/// measured cannot be judged, and one it measured cannot be dropped without changing what the others mean.
pub fn rejudge(run: &Run, schema: &Schema, policy: &Policy) -> anyhow::Result<Calibration> {
  let names = |names: Vec<&str>| names.join(", ");
  let fields: Vec<&str> = schema.fields().iter().map(|field| field.name.as_str()).collect();
  let quantities: Vec<&str> = schema.quantities().iter().map(|spec| spec.name.as_str()).collect();
  if fields != run.info.fields {
    bail!(
      "the run was scored with the fields [{}] and this configuration has [{}]; a change of fields changes the \
       measurements, so run `calibrate` again under it",
      run.info.fields.join(", "),
      names(fields)
    );
  }
  if quantities != run.info.quantities {
    bail!(
      "the run was scored with the quantities [{}] and this configuration has [{}]; run `calibrate` again under it",
      run.info.quantities.join(", "),
      names(quantities)
    );
  }
  let pairs = run
    .calibration
    .pairs
    .iter()
    .map(|scored| {
      let mut scored = scored.clone();
      scored.evaluation.rejudge(schema, policy);
      scored
    })
    .collect();
  Ok(summarize(pairs, run.calibration.input.clone()))
}

/// What makes a comparison a failure, for a script that gates on one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailOn {
  /// A recycle that was rejected is now accepted.
  NewFalseAccepts,
  /// Any verdict changed, in either direction.
  AnyFlip,
  /// The new run gets more pairs wrong than the old.
  ErrorsUp,
}

/// Why `diff` fails the gate `on`, or `None` if it passes.
pub fn failure(diff: &Diff, on: FailOn) -> Option<String> {
  let count = |flip: Flip| diff.flips.iter().filter(|flipped| flipped.flip == flip).count();
  match on {
    FailOn::NewFalseAccepts => {
      let accepts = count(Flip::NewFalseAccept);
      (accepts > 0).then(|| format!("{accepts} recycle(s) that were rejected are now accepted"))
    }
    FailOn::AnyFlip => (!diff.flips.is_empty()).then(|| format!("{} verdict(s) changed", diff.flips.len())),
    FailOn::ErrorsUp => (diff.new.errors() > diff.old.errors()).then(|| {
      format!(
        "the new run gets {} pair(s) wrong and the old one {}",
        diff.new.errors(),
        diff.old.errors()
      )
    }),
  }
}

fn verdict(accepted: bool) -> &'static str {
  if accepted { "accepted" } else { "rejected" }
}

fn describe_range(range: Option<Range>) -> String {
  range.map_or_else(
    || "-".to_owned(),
    |range| format!("{:.3} to {:.3} ({})", range.low, range.high, range.count),
  )
}

/// Renders a comparison as a report, for a terminal. `old` and `new` are one line each saying what the runs were.
pub fn render_diff(diff: &Diff, old: &str, new: &str) -> String {
  let mut out = String::new();
  // Writing to a `String` cannot fail, here and below.
  let _ = writeln!(out, "old    {old}\nnew    {new}");
  let _ = writeln!(
    out,
    "pairs  {} compared ({} only in the old run, {} only in the new)",
    diff.compared,
    diff.only_old.len(),
    diff.only_new.len()
  );

  if diff.flips.is_empty() {
    let _ = writeln!(out, "\nno verdict changed");
  } else {
    let _ = writeln!(
      out,
      "\n{:<8} {:>5} {:<9}    {:>5} {:<9} {:<17} pair",
      "label", "old", "", "new", "", "what"
    );
    for flipped in &diff.flips {
      let _ = writeln!(
        out,
        "{:<8} {:>5.3} {:<9} -> {:>5.3} {:<9} {:<17} {} {}",
        flipped.label,
        flipped.similarity.0,
        verdict(flipped.accepted.0),
        flipped.similarity.1,
        verdict(flipped.accepted.1),
        flipped.flip.headline(),
        flipped.id,
        flipped.note
      );
    }
  }

  let _ = writeln!(out, "\n{:<28} {:>7} {:>7}", "", "old", "new");
  let _ = writeln!(
    out,
    "{:<28} {:>7} {:>7}",
    "edits wrongly rejected", diff.old.false_rejects, diff.new.false_rejects
  );
  let _ = writeln!(
    out,
    "{:<28} {:>7} {:>7}",
    "recycles wrongly accepted", diff.old.false_accepts, diff.new.false_accepts
  );
  let part = |suggestion: Option<(f32, f32)>, pick: fn((f32, f32)) -> f32| {
    suggestion.map_or_else(|| "-".to_owned(), |found| format!("{:.3}", pick(found)))
  };
  let _ = writeln!(
    out,
    "{:<28} {:>7} {:>7}",
    "best overall threshold",
    part(diff.old.suggestion, |found| found.0),
    part(diff.new.suggestion, |found| found.0)
  );
  let _ = writeln!(
    out,
    "{:<28} {:>7} {:>7}",
    "its margin",
    part(diff.old.suggestion, |found| found.1),
    part(diff.new.suggestion, |found| found.1)
  );

  if !diff.slices.is_empty() {
    let _ = writeln!(out, "\nslices whose errors changed");
    for slice in &diff.slices {
      let _ = writeln!(
        out,
        "{} {}: wrongly rejected {} -> {}, wrongly accepted {} -> {}",
        slice.by.replace('_', " "),
        slice.value,
        slice.false_rejects.0,
        slice.false_rejects.1,
        slice.false_accepts.0,
        slice.false_accepts.1
      );
    }
  }
  if !diff.fields.is_empty() {
    let _ = writeln!(
      out,
      "\nfield ranges that moved (among the pairs that changed the field)"
    );
    for shift in &diff.fields {
      let _ = writeln!(
        out,
        "{} in {}s: {} -> {}",
        shift.field,
        shift.label.as_str(),
        describe_range(shift.old),
        describe_range(shift.new)
      );
    }
  }
  out
}

/// Writes a comparison as result events: one `flip` line per changed verdict, then a `diff` line with the totals.
pub fn emit(diff: &Diff) {
  for flipped in &diff.flips {
    tracing::info!(
      target: RESULT,
      pair_id = flipped.id.as_str(),
      label = flipped.label.as_str(),
      what = flipped.flip.as_str(),
      similarity_old = six(flipped.similarity.0),
      similarity_new = six(flipped.similarity.1),
      accepted_old = flipped.accepted.0,
      accepted_new = flipped.accepted.1,
      "flip"
    );
  }
  let count = |flip: Flip| diff.flips.iter().filter(|flipped| flipped.flip == flip).count();
  let threshold = |suggestion: Option<(f32, f32)>| suggestion.map(|found| six(found.0));
  tracing::info!(
    target: RESULT,
    compared = diff.compared,
    only_old = diff.only_old.len(),
    only_new = diff.only_new.len(),
    flips = diff.flips.len(),
    new_false_rejects = count(Flip::NewFalseReject),
    new_false_accepts = count(Flip::NewFalseAccept),
    fixed = count(Flip::Fixed),
    false_rejects_old = diff.old.false_rejects,
    false_rejects_new = diff.new.false_rejects,
    false_accepts_old = diff.old.false_accepts,
    false_accepts_new = diff.new.false_accepts,
    threshold_old = threshold(diff.old.suggestion),
    threshold_new = threshold(diff.new.suggestion),
    "diff"
  );
}

#[cfg(test)]
mod tests {
  use similarity_core::testing::tiny_embedder;
  use similarity_core::{EmbedderOptions, FieldSpec, QuantitySpec};

  use super::*;
  use crate::offline::events;
  use crate::offline::testing::{scored, scored_as};
  use crate::offline::{Offline, Selection, calibrate, parse_full_pairs};

  fn summarized(pairs: Vec<ScoredPair>) -> Calibration {
    summarize(pairs, Input::default())
  }

  /// A pair with its own id, so that the two runs can be matched up.
  fn pair(id: &str, label: Label, similarity: f32, accepted: bool) -> ScoredPair {
    scored_as(label, similarity, accepted, |pair| {
      pair.id = id.to_owned();
      pair.note = format!("note of {id}");
      pair
        .before
        .texts
        .insert("category".to_owned(), "Edibles > Gummies".to_owned());
    })
  }

  #[test]
  fn identical_runs_differ_in_nothing() {
    let run = summarized(vec![
      pair("p1", Label::Edit, 0.95, true),
      pair("p2", Label::Recycle, 0.2, false),
    ]);
    let diff = diff(&run, &run.clone());
    assert_eq!((diff.compared, diff.flips.len()), (2, 0));
    assert!(diff.only_old.is_empty() && diff.only_new.is_empty());
    assert!(diff.slices.is_empty() && diff.fields.is_empty());
    assert_eq!(diff.old, diff.new);
    for on in [FailOn::NewFalseAccepts, FailOn::AnyFlip, FailOn::ErrorsUp] {
      assert_eq!(failure(&diff, on), None);
    }
    let report = render_diff(&diff, "run a", "run b");
    assert!(report.contains("no verdict changed"), "{report}");
    assert!(
      report.contains("pairs  2 compared (0 only in the old run, 0 only in the new)"),
      "{report}"
    );
  }

  #[test]
  fn every_changed_verdict_is_named_for_what_it_means() {
    let old = summarized(vec![
      pair("edit-broken", Label::Edit, 0.92, true),
      pair("edit-fixed", Label::Edit, 0.70, false),
      pair("recycle-leaked", Label::Recycle, 0.60, false),
      pair("recycle-fixed", Label::Recycle, 0.85, true),
      pair("steady", Label::Edit, 0.99, true),
    ]);
    let new = summarized(vec![
      pair("edit-broken", Label::Edit, 0.50, false),
      pair("edit-fixed", Label::Edit, 0.90, true),
      pair("recycle-leaked", Label::Recycle, 0.91, true),
      pair("recycle-fixed", Label::Recycle, 0.80, false),
      pair("steady", Label::Edit, 0.98, true),
    ]);
    let diff = diff(&old, &new);

    let flips: Vec<(&str, Flip)> = diff
      .flips
      .iter()
      .map(|flipped| (flipped.id.as_str(), flipped.flip))
      .collect();
    assert_eq!(
      flips,
      [
        ("edit-broken", Flip::NewFalseReject),
        ("recycle-leaked", Flip::NewFalseAccept),
        ("edit-fixed", Flip::Fixed),
        ("recycle-fixed", Flip::Fixed),
      ],
      "largest change in score first"
    );
    assert_eq!((diff.old.false_rejects, diff.old.false_accepts), (1, 1));
    assert_eq!((diff.new.false_rejects, diff.new.false_accepts), (1, 1));

    assert!(
      failure(&diff, FailOn::NewFalseAccepts)
        .unwrap()
        .contains("1 recycle(s)")
    );
    assert!(failure(&diff, FailOn::AnyFlip).unwrap().contains("4 verdict(s)"));
    assert_eq!(failure(&diff, FailOn::ErrorsUp), None, "as many wrong as before");

    let report = render_diff(&diff, "a", "b");
    let squeezed: Vec<String> = report
      .lines()
      .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
      .collect();
    assert!(
      squeezed
        .contains(&"edit 0.920 accepted -> 0.500 rejected NEW FALSE REJECT edit-broken note of edit-broken".to_owned()),
      "{report}"
    );
    assert!(
      squeezed
        .contains(&"recycle 0.850 accepted -> 0.800 rejected fixed recycle-fixed note of recycle-fixed".to_owned()),
      "{report}"
    );
    assert!(squeezed.contains(&"edits wrongly rejected 1 1".to_owned()), "{report}");
  }

  #[test]
  fn more_errors_than_before_fails_the_gate_that_asks_for_that() {
    let old = summarized(vec![
      pair("p1", Label::Edit, 0.95, true),
      pair("p2", Label::Recycle, 0.2, false),
    ]);
    let new = summarized(vec![
      pair("p1", Label::Edit, 0.6, false),
      pair("p2", Label::Recycle, 0.2, false),
    ]);
    let diff = diff(&old, &new);
    assert!(
      failure(&diff, FailOn::ErrorsUp)
        .unwrap()
        .contains("1 pair(s) wrong and the old one 0")
    );
    assert_eq!(failure(&diff, FailOn::NewFalseAccepts), None);

    // The slice the new error is in is named.
    assert_eq!(
      diff.slices,
      [SliceChange {
        by: "category",
        value: "Edibles".to_owned(),
        false_rejects: (0, 1),
        false_accepts: (0, 0),
      }]
    );
    let report = render_diff(&diff, "a", "b");
    assert!(
      report.contains("category Edibles: wrongly rejected 0 -> 1, wrongly accepted 0 -> 0"),
      "{report}"
    );
  }

  #[test]
  fn pairs_only_one_run_has_are_counted_and_left_out_of_the_comparison() {
    let old = summarized(vec![
      pair("both", Label::Edit, 0.95, true),
      pair("gone", Label::Edit, 0.5, false),
    ]);
    let new = summarized(vec![
      pair("both", Label::Edit, 0.95, true),
      pair("arrived", Label::Recycle, 0.9, true),
    ]);
    let diff = diff(&old, &new);
    assert_eq!(diff.compared, 1);
    assert_eq!(
      (diff.only_old.as_slice(), diff.only_new.as_slice()),
      (&["gone".to_owned()][..], &["arrived".to_owned()][..])
    );
    assert_eq!(
      (diff.old.errors(), diff.new.errors()),
      (0, 0),
      "the errors of pairs that are not shared are nobody's regression"
    );
  }

  #[test]
  fn the_verdict_on_an_unlabeled_pair_changing_is_reported_as_neither_better_nor_worse() {
    let unsure = |accepted: bool| {
      scored_as(Label::Edit, 0.9, accepted, |pair| {
        pair.id = "u1".to_owned();
        pair.label = None;
        pair.label_text = "unsure".to_owned();
      })
    };
    let diff = diff(&summarized(vec![unsure(true)]), &summarized(vec![unsure(false)]));
    assert_eq!(diff.flips[0].flip, Flip::Unlabeled);
    assert_eq!(failure(&diff, FailOn::ErrorsUp), None);
    assert!(failure(&diff, FailOn::AnyFlip).is_some());
  }

  #[test]
  fn two_runs_are_over_the_same_pairs_only_if_file_and_selection_agree() {
    let input = |sha: &str, split: Option<&str>| Input {
      sha256: sha.to_owned(),
      split: split.map(str::to_owned),
      ..Input::default()
    };
    assert!(same_pairs(&input("aa", None), &input("aa", None)));
    assert!(!same_pairs(&input("aa", None), &input("bb", None)));
    assert!(!same_pairs(&input("aa", Some("holdout")), &input("aa", None)));
  }

  fn schema(name_weight: f32, name_floor: Option<f32>, weight_limit: Option<f64>) -> Schema {
    let field = |name: &str, weight: f32, reject_below: Option<f32>| FieldSpec {
      name: name.to_owned(),
      weight,
      required: false,
      reject_below,
    };
    Schema::new(vec![
      field("name", name_weight, name_floor),
      field("category", 2.0, None),
    ])
    .unwrap()
    .with_quantities(vec![QuantitySpec {
      name: "weight".to_owned(),
      reject_above_factor: weight_limit,
      reject_incomparable: false,
    }])
    .unwrap()
  }

  const PAIRS: &str = concat!(
    r#"{"pair_id": "p1", "label": "edit", "before": {"name": "blueberry gummies", "category": "edibles", "#,
    r#""weight": {"value": 1, "unit": "g"}}, "after": {"name": "blueberry gummies mg", "category": "edibles", "#,
    r#""weight": {"value": 3, "unit": "g"}}}"#,
    "\n",
    r#"{"pair_id": "p2", "label": "recycle", "before": {"name": "blueberry gummies", "category": "edibles", "#,
    r#""weight": {"value": 1, "unit": "g"}}, "after": {"name": "vape cartridge resin", "category": "flower", "#,
    r#""weight": {"value": 8, "unit": "g"}}}"#,
    "\n",
    r#"{"pair_id": "p3", "label": "edit", "before": {"name": "indica flower", "category": "flower"}, "#,
    r#""after": {"name": "sativa flower", "category": "flower"}}"#,
  );

  /// A run scored by the tiny encoder under a lenient configuration, as it was in memory and as its events read
  /// back.
  fn run() -> (Calibration, Run) {
    let options = EmbedderOptions {
      max_tokens: 16,
      batch_size: 8,
      output_dim: None,
    };
    let policy = Policy::new(-1.0, None).unwrap();
    let offline = Offline::new(tiny_embedder(&options), schema(1.0, None, None), policy);
    let pairs = parse_full_pairs(PAIRS, offline.schema(), Selection::default()).unwrap();
    let calibration = calibrate(&offline, pairs).unwrap();
    let captured = similarity_log::testing::capture("warn", || events::emit(&offline, &policy, &calibration, 0));
    let run = events::read(&captured.results.join("\n")).unwrap();
    (calibration, run)
  }

  #[test]
  fn a_run_judged_again_under_its_own_configuration_is_the_same_run() {
    let (_, run) = run();
    let again = rejudge(&run, &schema(1.0, None, None), &Policy::new(-1.0, None).unwrap()).unwrap();
    let diff = diff(&run.calibration, &again);
    assert_eq!((diff.compared, diff.flips.len()), (3, 0));
    assert_eq!(
      (diff.old.false_rejects, diff.old.false_accepts),
      (diff.new.false_rejects, diff.new.false_accepts)
    );
    // The overall scores are recomputed from field similarities written to six places, so they agree to about
    // that and not bit for bit.
    for (before, after) in run.calibration.pairs.iter().zip(&again.pairs) {
      assert!((before.similarity() - after.similarity()).abs() < 2e-6);
    }
    let (old, new) = (diff.old.suggestion.unwrap(), diff.new.suggestion.unwrap());
    assert!((old.0 - new.0).abs() < 2e-6 && (old.1 - new.1).abs() < 2e-6);
    assert!(diff.slices.is_empty() && diff.fields.is_empty());
  }

  #[test]
  fn judging_a_run_again_from_its_events_gives_what_judging_it_in_memory_gives() {
    let (in_memory, run) = run();
    // Everything a second configuration may change: a weight, a floor, the threshold, a quantity limit.
    let (other, strict) = (schema(9.0, Some(0.9999), Some(4.0)), Policy::new(0.2, None).unwrap());
    let from_events = rejudge(&run, &other, &strict).unwrap();

    for (found, original) in from_events.pairs.iter().zip(&in_memory.pairs) {
      let mut wanted = original.evaluation.clone();
      wanted.rejudge(&other, &strict);
      assert!(
        (found.similarity() - wanted.comparison.similarity).abs() < 2e-6,
        "{}: {} against {}",
        found.pair.id,
        found.similarity(),
        wanted.comparison.similarity
      );
      let codes =
        |reasons: &[similarity_core::Reason]| reasons.iter().map(similarity_core::Reason::code).collect::<Vec<_>>();
      assert_eq!(
        codes(&found.evaluation.reasons),
        codes(&wanted.reasons),
        "{}",
        found.pair.id
      );
      assert_eq!(found.evaluation.quantities, wanted.quantities);
    }
    // The recycle's weight went up eightfold, which the second configuration limits to four.
    let recycle = &from_events.pairs[1];
    assert!(recycle.evaluation.quantities[0].beyond_limit);
    assert!(
      recycle
        .evaluation
        .reasons
        .iter()
        .any(|reason| reason.code() == "quantity_beyond_factor")
    );
    // The edit whose name changed fails the new floor on the name, and is now an error.
    assert!(from_events.false_rejects >= 1);
    let diff = diff(&run.calibration, &from_events);
    assert!(diff.flips.iter().any(|flipped| flipped.flip == Flip::NewFalseReject));
    assert!(
      diff.flips.iter().any(|flipped| flipped.flip == Flip::Fixed),
      "the recycle is now rejected"
    );
  }

  #[test]
  fn a_configuration_with_other_fields_or_quantities_is_refused() {
    let (_, run) = run();
    let policy = Policy::new(0.5, None).unwrap();
    let fewer = Schema::new(vec![FieldSpec {
      name: "name".to_owned(),
      weight: 1.0,
      required: false,
      reject_below: None,
    }])
    .unwrap();
    let error = rejudge(&run, &fewer, &policy).unwrap_err().to_string();
    assert!(
      error.contains("[category, name]") && error.contains("[name]") && error.contains("run `calibrate` again"),
      "{error}"
    );

    let no_quantities = Schema::new(schema(1.0, None, None).fields().to_vec()).unwrap();
    let error = rejudge(&run, &no_quantities, &policy).unwrap_err().to_string();
    assert!(error.contains("quantities [weight]"), "{error}");
  }

  #[test]
  fn a_comparison_is_also_written_as_events() {
    let old = summarized(vec![
      pair("p1", Label::Edit, 0.95, true),
      scored(Label::Recycle, 0.2, false),
    ]);
    let new = summarized(vec![
      pair("p1", Label::Edit, 0.6, false),
      scored(Label::Recycle, 0.2, false),
    ]);
    let diff = diff(&old, &new);
    let captured = similarity_log::testing::capture("warn", || emit(&diff));
    let events: Vec<serde_json::Value> = captured
      .results
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();
    assert_eq!(events.len(), 2);
    assert_eq!(
      (&events[0]["message"], &events[0]["pair_id"], &events[0]["what"]),
      (&"flip".into(), &"p1".into(), &"new_false_reject".into())
    );
    assert_eq!(
      (&events[0]["accepted_old"], &events[0]["accepted_new"]),
      (&true.into(), &false.into())
    );
    let summary = &events[1];
    assert_eq!(summary["message"], "diff");
    assert_eq!(
      (
        &summary["compared"],
        &summary["flips"],
        &summary["new_false_rejects"],
        &summary["false_rejects_new"]
      ),
      (&2.into(), &1.into(), &1.into(), &1.into())
    );
  }
}
