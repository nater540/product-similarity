//! The committed verdict snapshot, checked with no model (D22).
//!
//! `data/regression.snapshot.ndjson` holds the similarities the pinned model gave the curated cases of
//! `data/regression.jsonl`. Most changes that could move a verdict do not change those similarities: a weight, a
//! floor, a threshold, a quantity limit, or the code that judges. So the snapshot is judged again here, under the
//! `config.example.toml` and the scoring code of this build, and nothing about any case may come out different.
//! That runs on every machine in a fraction of a second. A change to the similarities themselves is caught by
//! `tests/real_snapshot.rs`, which needs the model.

#![allow(
  clippy::unwrap_used,
  clippy::expect_used,
  clippy::panic,
  clippy::indexing_slicing,
  missing_docs
)]

use similarity_server::config::Config;
use similarity_server::offline::diff::{self, TOLERANCE};
use similarity_server::offline::events::{self, Run};
use similarity_server::offline::{Selection, parse_full_pairs};

const CASES: &str = include_str!("../../../data/regression.jsonl");
const SNAPSHOT: &str = include_str!("../../../data/regression.snapshot.ndjson");

fn config() -> Config {
  Config::parse(include_str!("../../../config.example.toml")).unwrap()
}

fn snapshot() -> Run {
  events::read(SNAPSHOT).unwrap()
}

#[test]
fn the_snapshot_was_made_from_the_committed_cases() {
  let cases = parse_full_pairs(CASES, &config().schema().unwrap(), Selection::default()).unwrap();
  let snapshot = snapshot();
  assert_eq!(
    snapshot.calibration.input.sha256, cases.input.sha256,
    "data/regression.jsonl was edited after the snapshot was made; run `just snapshot` (it needs the model)"
  );
  let ids = |ids: Vec<&str>| {
    let mut ids: Vec<String> = ids.into_iter().map(str::to_owned).collect();
    ids.sort();
    ids
  };
  assert_eq!(
    ids(
      snapshot
        .calibration
        .pairs
        .iter()
        .map(|scored| scored.pair.id.as_str())
        .collect()
    ),
    ids(cases.pairs.iter().map(|pair| pair.id.as_str()).collect())
  );
}

#[test]
fn every_curated_case_fits_the_example_schema_and_has_an_id() {
  let cases = parse_full_pairs(CASES, &config().schema().unwrap(), Selection::default()).unwrap();
  assert!(cases.pairs.len() >= 20, "{} cases", cases.pairs.len());
  assert!(
    cases.input.ignored.is_empty(),
    "a key the schema cannot compare is a case that does not test what it says: {:?}",
    cases.input.ignored
  );
  for pair in &cases.pairs {
    assert!(
      pair.id.starts_with("r-"),
      "`{}` should carry a `pair_id` of its own",
      pair.id
    );
    assert!(!pair.note.is_empty(), "`{}` has no note", pair.id);
  }
  assert!(
    cases.pairs.iter().any(|pair| pair.anchor.is_some()),
    "the two-step swap is among them"
  );
}

#[test]
fn the_snapshot_judged_again_under_the_example_config_changes_nothing() {
  let (config, snapshot) = (config(), snapshot());
  let judged = diff::rejudge(&snapshot, &config.schema().unwrap(), &config.policy().unwrap()).unwrap();
  let difference = diff::diff(&snapshot.calibration, &judged, TOLERANCE);
  assert!(
    difference.flips.is_empty() && difference.moved.is_empty(),
    "`config.example.toml` or the scoring code now judges a curated case differently. If that is intended, run \
     `just snapshot` (it needs the model) and put what it prints in the commit message.\n{}",
    diff::render_diff(
      &difference,
      "the committed snapshot",
      "the snapshot judged again by this build"
    )
  );
}

#[test]
fn the_snapshot_holds_no_timestamp() {
  for line in SNAPSHOT.lines() {
    let object: serde_json::Map<String, serde_json::Value> = serde_json::from_str(line).unwrap();
    assert!(!object.contains_key("timestamp"), "{line}");
  }
}
