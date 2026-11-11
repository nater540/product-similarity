//! The curated hard cases, scored with the real model and held to the committed snapshot (D22).
//!
//! `data/regression.jsonl` is a set of synthetic pairs chosen to be hard: unit rewrites, same-brand variants, a
//! blank-and-refill swap, look-alike characters. `data/regression.snapshot.ndjson` is what the pinned model and
//! `config.example.toml` made of them, as a `calibrate --ndjson` run with its timestamps removed. This test scores
//! the cases afresh and fails if a verdict flipped, a score moved by more than the tolerance, or a reason or a kind
//! of change is not what it was. It catches what `tests/regression_snapshot.rs` cannot: a change to what the
//! similarities are (the model files, the encoder, the loader, the tokenizer, pooling, truncation).
//!
//! After an intended change, `just snapshot` runs this with `UPDATE_SNAPSHOT=1`: it prints what moved, refuses to
//! pin a case that sits on a threshold, and rewrites the snapshot. The same file checks that the calibration
//! reports the README quotes are what `calibrate` prints.
//!
//! Ignored by default because it needs the 90 MB checkpoint: `just fetch-model && just test-snapshot`.

#![allow(
  clippy::unwrap_used,
  clippy::expect_used,
  clippy::panic,
  clippy::indexing_slicing,
  clippy::print_stdout,
  missing_docs
)]

use std::path::PathBuf;

use similarity_core::{Device, Embedder, Policy};
use similarity_server::config::Config;
use similarity_server::offline::diff::{self, TOLERANCE};
use similarity_server::offline::events::{self, Run};
use similarity_server::offline::pairs::Confidence;
use similarity_server::offline::{Offline, Selection, calibrate, parse_full_pairs, parse_pairs, render_calibration};

fn root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The shipped configuration on the pinned model.
fn scorer() -> (Offline, Policy) {
  let mut config = Config::parse(include_str!("../../../config.example.toml")).unwrap();
  config.model.dir = root().join(&config.model.dir);
  let embedder = Embedder::load(&config.model.dir, Device::flex(), &config.model.embedder_options()).unwrap();
  let policy = config.policy().unwrap();
  (Offline::new(embedder, config.schema().unwrap(), policy), policy)
}

/// One event line without its timestamp, so that writing the snapshot again changes only the lines that moved.
fn undated(line: &str) -> String {
  let mut object: serde_json::Map<String, serde_json::Value> = serde_json::from_str(line).unwrap();
  object.remove("timestamp");
  serde_json::Value::Object(object).to_string()
}

#[test]
#[ignore = "needs ./models/mdbr-leaf-mt (`just fetch-model`); run with `just test-snapshot`"]
fn the_pinned_model_gives_the_committed_verdicts_and_scores() {
  let (offline, policy) = scorer();
  let cases = std::fs::read_to_string(root().join("data/regression.jsonl")).unwrap();
  let pairs = parse_full_pairs(&cases, offline.schema(), Selection::default()).unwrap();
  let calibration = calibrate(&offline, pairs).unwrap();
  // Through the events, as the committed run went, so that both sides are rounded alike.
  let captured = similarity_log::testing::capture("warn", || events::emit(&offline, &policy, &calibration, 0));
  let lines: Vec<String> = captured.results.iter().map(|line| undated(line)).collect();
  let fresh = events::read(&lines.join("\n")).unwrap();

  let path = root().join("data/regression.snapshot.ndjson");
  let committed: Option<Run> = std::fs::read_to_string(&path)
    .ok()
    .map(|text| events::read(&text).unwrap());
  let compared = committed.as_ref().map(|old| {
    let difference = diff::diff(&old.calibration, &fresh.calibration, TOLERANCE);
    let report = diff::render_diff(&difference, "the committed snapshot", "this build");
    (difference, report)
  });

  if std::env::var_os("UPDATE_SNAPSHOT").is_some() {
    match &compared {
      Some((_, report)) => println!("{report}"),
      None => println!("there was no snapshot; writing the first one"),
    }
    let borders = diff::borderline(&fresh.calibration, offline.schema(), &policy, TOLERANCE);
    assert!(
      borders.is_empty(),
      "these cases sit within {TOLERANCE} of a limit they are judged against, and would flip on the noise between \
       two machines; change the case, not the snapshot: {borders:#?}"
    );
    std::fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();
    println!("wrote {}", path.display());
    return;
  }

  let committed = committed.expect("data/regression.snapshot.ndjson is missing; write it with `just snapshot`");
  let (difference, report) = compared.unwrap();
  assert_eq!(
    committed.info.fingerprint, fresh.info.fingerprint,
    "the snapshot was made with another model; a new model needs a new snapshot (`just snapshot`)"
  );
  assert_eq!(
    committed.calibration.input.sha256, fresh.calibration.input.sha256,
    "data/regression.jsonl changed since the snapshot was made; run `just snapshot`"
  );
  assert!(
    difference.only_old.is_empty() && difference.only_new.is_empty(),
    "the snapshot and the cases are not the same pairs\n{report}"
  );
  assert!(
    difference.flips.is_empty() && difference.moved.is_empty(),
    "the curated cases no longer score as pinned. If that is intended, run `just snapshot` and put what it prints \
     in the commit message.\n{report}"
  );
}

/// The fenced block that follows each `<!-- measured: calibrate FILE [--full] -->` marker in the README, with the
/// file and whether it is read in the full format.
fn quoted_reports(readme: &str) -> Vec<(String, bool, Vec<String>)> {
  let mut found = Vec::new();
  let mut lines = readme.lines();
  while let Some(line) = lines.next() {
    let Some(command) = line
      .trim()
      .strip_prefix("<!-- measured: calibrate ")
      .and_then(|rest| rest.strip_suffix(" -->"))
    else {
      continue;
    };
    let mut words = command.split_whitespace();
    let file = words.next().unwrap().to_owned();
    let full = words.any(|word| word == "--full");
    let block: Vec<String> = lines
      .by_ref()
      .skip_while(|line| !line.starts_with("```"))
      .skip(1)
      .take_while(|line| !line.starts_with("```"))
      .map(str::to_owned)
      .collect();
    found.push((file, full, block));
  }
  found
}

/// CLAUDE.md: every number in the README was measured. For the calibration reports it quotes, this is the check.
#[test]
#[ignore = "needs ./models/mdbr-leaf-mt (`just fetch-model`); run with `just test-snapshot`"]
fn the_calibration_reports_quoted_in_the_readme_are_what_calibrate_prints() {
  let (offline, policy) = scorer();
  let readme = std::fs::read_to_string(root().join("README.md")).unwrap();
  let quoted = quoted_reports(&readme);
  assert!(
    quoted.len() >= 2,
    "the README marks its quoted reports with `<!-- measured: calibrate ... -->`"
  );

  for (file, full, block) in quoted {
    let text = std::fs::read_to_string(root().join(&file)).unwrap();
    let pairs = if full {
      // What the command does when asked for nothing else.
      let selection = Selection {
        split: None,
        min_confidence: Some(Confidence::Medium),
      };
      parse_full_pairs(&text, offline.schema(), selection)
    } else {
      parse_pairs(&text)
    }
    .unwrap();
    let calibration = calibrate(&offline, pairs).unwrap();
    let report = format!(
      "{}\n{}",
      offline.model_line(),
      render_calibration(&calibration, &policy, 0)
    );
    let printed: Vec<&str> = report.lines().map(str::trim_end).collect();
    for line in &block {
      let line = line.trim_end();
      // `...` stands for lines left out, and a `$` line is the command, not its output.
      if line.is_empty() || line == "..." || line.starts_with("$ ") {
        continue;
      }
      assert!(
        printed.contains(&line),
        "the README quotes a line that `calibrate {file}` does not print:\n{line}\n\nit prints:\n{report}"
      );
    }
  }
}
