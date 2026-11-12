//! Two recorded runs, side by side, as text.
//!
//! Two calibrations are compared as `calibrate-diff` compares them: this adds only where each came from. Two
//! training runs are laid out as what went in (the base model, the data, the options) and what came out (the
//! losses, the separation), with every row that differs marked, so that the one thing changed between two runs
//! is the first thing seen.

use std::fmt::Write as _;

use anyhow::bail;
use serde_json::Value;
use similarity_runs::Listed;

use super::Loaded;
use super::training::{SeparationLine, StepLine, TrainingRun};
use crate::offline::diff;

/// One line saying which run this is: its id, how it stands, and what built the program that ran it.
pub fn describe(listed: &Listed) -> String {
  let build = &listed.manifest.build;
  let dirty = if build.dirty == Some(true) {
    " (uncommitted changes)"
  } else {
    ""
  };
  format!(
    "{}   {}   commit {}{dirty}",
    listed.manifest.id,
    listed.status.as_str(),
    build.commit
  )
}

fn plain(value: &Value) -> String {
  match value {
    Value::String(text) => text.clone(),
    Value::Array(items) => items.iter().map(plain).collect::<Vec<_>>().join(","),
    Value::Null => "-".to_owned(),
    other => other.to_string(),
  }
}

/// A short form of a digest or hash: enough to tell two apart.
fn short(text: &str) -> String {
  text.chars().take(12).collect()
}

fn number(value: Option<f64>, places: usize) -> String {
  value.map_or_else(|| "-".to_owned(), |value| format!("{value:.places$}"))
}

/// The rows that say what went into a training run: the model it started from, its data, and its options.
pub fn inputs(listed: &Listed) -> Vec<(String, String)> {
  let manifest = &listed.manifest;
  let mut rows = Vec::new();
  if let Some(model) = &manifest.model {
    rows.push(("base model".to_owned(), short(&model.digest)));
  }
  for input in &manifest.inputs {
    let examples = input
      .examples
      .map_or_else(String::new, |count| format!(", {count} examples"));
    rows.push((input.role.clone(), format!("{}{examples}", short(&input.sha256))));
  }
  if let Some(Value::Object(options)) = &manifest.options {
    rows.extend(options.iter().map(|(name, value)| (name.clone(), plain(value))));
  }
  rows
}

/// The rows that say what a training run produced.
pub fn outcomes(run: &TrainingRun) -> Vec<(String, String)> {
  let pair: fn(&StepLine) -> Option<f64> = |line| line.pair_loss;
  let group: fn(&StepLine) -> Option<f64> = |line| line.group_loss;
  let lowest = |loss| {
    run
      .lowest(loss)
      .map_or_else(|| "-".to_owned(), |(value, step)| format!("{value:.4} at {step}"))
  };
  let mut rows = vec![
    (
      "steps".to_owned(),
      run
        .finished
        .as_ref()
        .map_or_else(|| "-".to_owned(), |finished| finished.steps.to_string()),
    ),
    ("last pair loss".to_owned(), number(run.last(pair), 4)),
    ("lowest pair loss".to_owned(), lowest(pair)),
    ("last group loss".to_owned(), number(run.last(group), 4)),
  ];
  if let Some(finished) = &run.finished
    && let (Some(step), Some(auc)) = (finished.best_step, finished.best_auc)
  {
    rows.push((
      "best model (validation)".to_owned(),
      format!("step {step}, AUC {auc:.4}"),
    ));
  }
  if let Some(error) = &run.failure {
    rows.push(("failed".to_owned(), error.clone()));
  }
  rows
}

/// The measurement of a run on `split` after training: the final model's, or the chosen model's where only that
/// was measured (D24).
fn after<'a>(run: &'a TrainingRun, split: &str, width: usize) -> Option<&'a SeparationLine> {
  run
    .separation_at(split, "after", width)
    .or_else(|| run.separation_at(split, "selected", width))
}

/// One number of a separation line.
type Pick = fn(&SeparationLine) -> Option<f64>;

fn render_training(a: (&Listed, &TrainingRun), b: (&Listed, &TrainingRun), out: &mut String) {
  // Writing to a `String` cannot fail, here and below.
  let _ = writeln!(out, "a      {}\nb      {}\n", describe(a.0), describe(b.0));
  let _ = writeln!(out, "  {:<26} {:<28} b", "", "a");
  let mut differing = 0_usize;
  for (left, right) in [(inputs(a.0), inputs(b.0)), (outcomes(a.1), outcomes(b.1))] {
    let mut names: Vec<&String> = left.iter().chain(&right).map(|(name, _)| name).collect();
    // In the order first seen, each once: the manifest's own order, which a person recognizes.
    let mut seen = std::collections::BTreeSet::new();
    names.retain(|name| seen.insert(*name));
    for name in names {
      let of = |rows: &[(String, String)]| {
        rows
          .iter()
          .find(|(found, _)| found == name)
          .map_or("-", |(_, value)| value.as_str())
          .to_owned()
      };
      let (ours, theirs) = (of(&left), of(&right));
      let mark = if ours == theirs {
        ' '
      } else {
        differing = differing.saturating_add(1);
        '*'
      };
      let _ = writeln!(out, "{mark} {name:<26} {ours:<28} {theirs}");
    }
    let _ = writeln!(out);
  }
  let _ = writeln!(out, "* differs between the two ({differing} row(s))");

  // At the narrowest width both runs were measured at: the one a server is most likely run at, and the one the
  // trainer compares models at unless told otherwise. The other widths are in each run's events.
  let widths: Vec<usize> = a
    .1
    .widths()
    .into_iter()
    .filter(|width| b.1.widths().contains(width))
    .collect();
  for split in ["validation", "held_out", "training"] {
    for width in widths.iter().take(1) {
      let lines = [
        a.1.separation_at(split, "before", *width),
        after(a.1, split, *width),
        b.1.separation_at(split, "before", *width),
        after(b.1, split, *width),
      ];
      if lines.iter().all(Option::is_none) {
        continue;
      }
      // Where only the chosen model was measured after training (D24), the column says so.
      let stage = |line: Option<&SeparationLine>| match line.map(|line| line.stage.as_str()) {
        Some("selected") => "chosen",
        _ => "after",
      };
      let _ = writeln!(
        out,
        "\n{:<28} {:>9} {:>9} {:>9} {:>9}",
        format!("{split}, width {width}"),
        "a before",
        format!("a {}", stage(lines.get(1).copied().flatten())),
        "b before",
        format!("b {}", stage(lines.get(3).copied().flatten()))
      );
      let rows: [(&str, Pick); 4] = [
        ("edits mean", |line| line.edits_mean),
        ("recycles mean", |line| line.recycles_mean),
        ("gap of means", |line| line.gap_of_means),
        ("AUC", |line| line.auc),
      ];
      for (name, pick) in rows {
        let cells: Vec<String> = lines.iter().map(|line| number(line.and_then(pick), 4)).collect();
        let _ = writeln!(
          out,
          "{name:<28} {}",
          cells
            .iter()
            .map(|cell| format!("{cell:>9}"))
            .collect::<Vec<_>>()
            .join(" ")
        );
      }
      let starts = |line: Option<&SeparationLine>| line.and_then(|line| line.gap_of_means);
      if let (Some(ours), Some(theirs)) = (starts(lines[0]), starts(lines[2]))
        && (ours - theirs).abs() > 1e-9
      {
        let _ = writeln!(
          out,
          "the two did not start from the same model and data on this split, so their `after` columns are not like \
           for like"
        );
      }
    }
  }
}

/// Compares two loaded runs, as text for a terminal.
///
/// `intersect` and `tolerance` are for two calibrations, and mean what they mean to `calibrate-diff`.
///
/// # Errors
///
/// Returns an error if one run is a training run and the other a calibration, or if two calibrations were not
/// made from the same selection of the same pairs and `intersect` is not set.
pub fn compare(a: &Loaded, b: &Loaded, intersect: bool, tolerance: f32) -> anyhow::Result<String> {
  let mut out = String::new();
  match (a, b) {
    (
      Loaded::Training { chain: ours, run: left },
      Loaded::Training {
        chain: theirs,
        run: right,
      },
    ) => {
      let (Some(ours), Some(theirs)) = (ours.last(), theirs.last()) else {
        bail!("a training run with no directory cannot be compared");
      };
      render_training((ours, left), (theirs, right), &mut out);
    }
    (
      Loaded::Calibration {
        listed: ours,
        run: left,
      },
      Loaded::Calibration {
        listed: theirs,
        run: right,
      },
    ) => {
      if !intersect && !diff::same_pairs(&left.calibration.input, &right.calibration.input) {
        bail!(
          "the two calibrations were not made from the same selection of the same pairs file; pass --intersect to \
           compare the pairs they share"
        );
      }
      let difference = diff::diff(&left.calibration, &right.calibration, tolerance);
      out.push_str(&diff::render_diff(&difference, &describe(ours), &describe(theirs)));
    }
    _ => {
      let kind = |loaded: &Loaded| match loaded {
        Loaded::Training { .. } => "a training run",
        Loaded::Calibration { .. } => "a calibration",
      };
      bail!(
        "`{}` is {} and `{}` is {}; two runs of one kind can be compared",
        a.id(),
        kind(a),
        b.id(),
        kind(b)
      );
    }
  }
  Ok(out)
}

#[cfg(test)]
mod tests {
  use std::path::{Path, PathBuf};

  use similarity_core::provenance::Stamp;
  use similarity_runs::{
    Continues, EVENTS_FILE, Input, KIND_CALIBRATE, KIND_TRAIN, MANIFEST_FILE, ModelRef, Run, Spec,
  };

  use super::*;
  use crate::runs::{html, load};

  /// Records a run under `root` with `events` as what it emitted, and returns its directory. The run is left as
  /// a process that is still alive would leave it, which is all a reader needs.
  fn recorded(root: &Path, kind: &'static str, slug: &str, events: &str, continues: Option<Continues>) -> PathBuf {
    let run = Run::start(Spec {
      root,
      kind,
      slug: slug.to_owned(),
      build: Stamp {
        tool: "test".to_owned(),
        version: "0.1.0".to_owned(),
        commit: "0123456789ab".to_owned(),
        dirty: Some(false),
      },
      argv: Vec::new(),
      continues,
    })
    .unwrap();
    if kind == KIND_TRAIN {
      run.note_options(serde_json::json!({"epochs": 2, "learning_rate": 0.000_02, "widths": [1024, 256]}));
      run.note_model(ModelRef {
        role: "base".to_owned(),
        dir: "models/base".to_owned(),
        digest: "f46e48ee0ea2dd02".to_owned(),
        fingerprint: None,
      });
      run.note_input(Input {
        role: "pairs".to_owned(),
        path: "pairs.jsonl".to_owned(),
        sha256: "bedd362ac75d0000".to_owned(),
        examples: Some(34),
      });
    }
    let dir = run.dir().to_owned();
    std::fs::write(dir.join(EVENTS_FILE), events).unwrap();
    run.finish(Ok(())).unwrap();
    dir
  }

  /// A training run of `losses.len()` logged steps, ten apart, starting after `continues`.
  fn training(root: &Path, slug: &str, learning_rate: f64, losses: &[f64], continues: Option<Continues>) -> PathBuf {
    let start = continues.as_ref().map_or(0, |continues| continues.step);
    let last = start + losses.len() * 10;
    let separation = |split: &str, stage: &str, step: usize, width: usize, gap: f64, auc: f64| {
      format!(
        r#"{{"message":"separation","split":"{split}","stage":"{stage}","step":{step},"width":{width},"edits_mean":0.93,"recycles_mean":0.6,"gap_of_means":{gap},"auc":{auc}}}"#
      )
    };
    let mut events = vec![
      separation("validation", "before", start, 256, 0.3, 0.9),
      separation("validation", "before", start, 1024, 0.29, 0.91),
    ];
    for (index, loss) in losses.iter().enumerate() {
      let step = start + (index + 1) * 10;
      events.push(format!(
        r#"{{"message":"training","step":{step},"of":40,"pair_loss":{loss},"grad_norm":1.0,"rate":{learning_rate}}}"#
      ));
    }
    events.push(separation("validation", "after", last, 256, 0.38, 0.97));
    events.push(separation("validation", "after", last, 1024, 0.37, 0.96));
    events.push(separation("held_out", "selected", last, 256, 0.44, 0.98));
    events.push(format!(
      r#"{{"message":"finished","out":"models/{slug}","steps":{last},"elapsed_s":12.5,"best_step":{last},"best_model":"models/{slug}","best_auc":0.97}}"#
    ));
    let dir = recorded(root, KIND_TRAIN, slug, &events.join("\n"), continues);
    // The option that differs between two runs of a comparison.
    let path = dir.join(MANIFEST_FILE);
    let mut manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["options"]["learning_rate"] = learning_rate.into();
    std::fs::write(&path, manifest.to_string()).unwrap();
    dir
  }

  #[test]
  fn two_training_runs_are_laid_side_by_side_with_what_differs_marked() {
    let root = tempfile::tempdir().unwrap();
    let a = load(&training(root.path(), "slow", 0.000_02, &[0.05, 0.03], None)).unwrap();
    let b = load(&training(root.path(), "fast", 0.000_06, &[0.05, 0.02], None)).unwrap();
    let text = compare(&a, &b, false, 0.001).unwrap();

    assert!(text.contains("-train-slow") && text.contains("-train-fast"), "{text}");
    assert!(
      text.contains("* learning_rate"),
      "the one option that was changed: {text}"
    );
    assert!(
      text.contains("  epochs") && text.contains("  base model") && text.contains("  pairs"),
      "{text}"
    );
    assert!(text.contains("1024,256"), "a list is written joined: {text}");
    assert!(
      text.contains("* lowest pair loss") && text.contains("0.0200 at 20"),
      "{text}"
    );
    assert!(
      text.contains("best model (validation)") && text.contains("AUC 0.9700"),
      "{text}"
    );
    // One width, the narrowest both have; the held-out column is named for what was measured.
    assert!(
      text.contains("validation, width 256") && !text.contains("width 1024"),
      "{text}"
    );
    assert!(
      text.contains("held_out, width 256") && text.contains("a chosen"),
      "{text}"
    );
  }

  /// A resumed run is two directories. Naming the second loads both, as one curve.
  #[test]
  fn a_resumed_run_is_loaded_with_the_run_it_continues() {
    let root = tempfile::tempdir().unwrap();
    let first = training(root.path(), "ft", 0.000_02, &[0.05, 0.04, 0.035], None);
    let first_id = similarity_runs::open(&first).unwrap().manifest.id;
    let continues = Continues {
      run: Some(first_id.clone()),
      from: "ft.checkpoints/step-000020".to_owned(),
      step: 20,
    };
    let second = training(root.path(), "ft", 0.000_02, &[0.03, 0.02], Some(continues));
    let Loaded::Training { chain, run } = load(&second).unwrap() else {
      panic!("a training run")
    };
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].manifest.id, first_id);
    let curve: Vec<(usize, Option<f64>)> = run.steps.iter().map(|line| (line.step, line.pair_loss)).collect();
    assert_eq!(
      curve,
      [(10, Some(0.05)), (20, Some(0.04)), (30, Some(0.03)), (40, Some(0.02))]
    );
    assert_eq!(run.resumed_at, [20]);

    // With the earlier directory gone, the link is an error that names it, not a shorter curve.
    std::fs::remove_dir_all(&first).unwrap();
    let error = format!("{:#}", load(&second).unwrap_err());
    assert!(error.contains(&first_id), "{error}");
  }

  #[test]
  fn a_training_run_and_a_calibration_are_not_compared_and_a_page_takes_both() {
    let root = tempfile::tempdir().unwrap();
    let trained = load(&training(root.path(), "ft", 0.000_02, &[0.05, 0.04], None)).unwrap();
    let calibration = concat!(
      r#"{"message":"calibration","fingerprint":"f46e48ee0ea2dd02-d256-t256","dimension":256,"max_tokens":256,"threshold":0.9,"pairs_sha256":"aa","pairs":2,"seed":0,"target":"result"}"#,
      "\n",
      r#"{"message":"schema_field","field":"name","weight":1.0,"required":true,"target":"result"}"#,
      "\n",
      r#"{"message":"pair","pair_id":"p1","label":"edit","similarity":0.95,"accepted":true,"wrong":false,"measure_changed":false,"target":"result"}"#,
      "\n",
      r#"{"message":"pair","pair_id":"p2","label":"recycle","similarity":0.4,"accepted":false,"wrong":false,"measure_changed":false,"target":"result"}"#,
    );
    let calibrated = load(&recorded(root.path(), KIND_CALIBRATE, "pairs", calibration, None)).unwrap();
    let error = compare(&trained, &calibrated, false, 0.001).unwrap_err().to_string();
    assert!(
      error.contains("a training run") && error.contains("a calibration"),
      "{error}"
    );

    // Two calibrations are compared as `calibrate-diff` compares them, under their run ids.
    let text = compare(&calibrated, &calibrated, false, 0.001).unwrap();
    assert!(
      text.contains("-calibrate-pairs") && text.contains("no verdict changed"),
      "{text}"
    );

    // The page takes any mix: a curve for the one, a histogram for the other, a table under each chart.
    let page = html::report(&[trained, calibrated]);
    assert!(page.starts_with("<!doctype html>") && page.trim_end().ends_with("</html>"));
    assert_eq!(
      page.matches("<svg").count(),
      5,
      "loss, gradient norm, rate, validation AUC, and one histogram"
    );
    assert_eq!(page.matches("The numbers").count(), 5, "every chart has its table");
    assert!(
      page.contains("suggested") && page.contains("threshold 0.900"),
      "both thresholds are named on the histogram"
    );
    assert!(!page.contains("<script"), "nothing runs");
    assert!(page.contains("learning_rate") && page.contains("AUC 0.9700"));
  }
}
