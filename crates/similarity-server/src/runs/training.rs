//! A training run read back from the events it left behind.
//!
//! The trainer writes progress as log lines (`training`, `checkpoint`) and results as result lines (`separation`,
//! `finished`). A recorded run keeps both in its `events.ndjson` (D23). This reads them back by message name.
//! Lines it does not know are skipped, and a field it does not use is ignored: events gain fields over time and
//! never repurpose one (D17).
//!
//! Every number a comparison or a report shows about a training run is derived here, and nowhere else.

use std::collections::BTreeMap;

use anyhow::{Context, bail};
use serde_json::{Map, Value};

/// One `training` line: the losses and the learning rate over the steps since the line before.
#[derive(Debug, Clone, PartialEq)]
pub struct StepLine {
  /// The step the line was written at.
  pub step: usize,
  /// How many steps the run has.
  pub of: usize,
  /// The mean loss on labeled pairs since the last line, if any were trained on.
  pub pair_loss: Option<f64>,
  /// The mean loss on catalog groups since the last line, if any were trained on.
  pub group_loss: Option<f64>,
  /// The mean gradient norm.
  pub grad_norm: Option<f64>,
  /// The learning rate.
  pub rate: Option<f64>,
}

/// One `separation` line: how a model separates edits from recycles on one split at one width.
#[derive(Debug, Clone, PartialEq)]
pub struct SeparationLine {
  /// `training`, `held_out`, or `validation`.
  pub split: String,
  /// `before`, `after`, `checkpoint`, or `selected`.
  pub stage: String,
  /// The step the model is from. Absent in events written before steps were recorded.
  pub step: Option<usize>,
  /// The embedding width.
  pub width: usize,
  /// The mean similarity of edits.
  pub edits_mean: Option<f64>,
  /// The mean similarity of recycles.
  pub recycles_mean: Option<f64>,
  /// The first minus the second.
  pub gap_of_means: Option<f64>,
  /// How often an edit scores above a recycle.
  pub auc: Option<f64>,
}

/// The `finished` line.
#[derive(Debug, Clone, PartialEq)]
pub struct Finished {
  /// Where the final model was written.
  pub out: String,
  /// How many steps the run took.
  pub steps: usize,
  /// How long the training took, in seconds.
  pub elapsed_s: Option<f64>,
  /// The step of the model validation chose, if models were compared.
  pub best_step: Option<usize>,
  /// Where that model is.
  pub best_model: Option<String>,
  /// Its AUC on the validation pairs.
  pub best_auc: Option<f64>,
}

/// A training run, from one process or stitched from the processes of a resumed one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainingRun {
  /// One line per logged step, in step order.
  pub steps: Vec<StepLine>,
  /// The steps a checkpoint was written at.
  pub checkpoints: Vec<usize>,
  /// The steps at which a later process went on from a checkpoint.
  pub resumed_at: Vec<usize>,
  /// Every separation line, in the order written.
  pub separation: Vec<SeparationLine>,
  /// The closing line, if the run got that far.
  pub finished: Option<Finished>,
  /// Why the run failed, if it said.
  pub failure: Option<String>,
}

struct Event<'a>(&'a Map<String, Value>);

impl Event<'_> {
  fn number(&self, key: &str) -> Option<f64> {
    self.0.get(key).and_then(Value::as_f64)
  }

  fn count(&self, key: &str) -> Option<usize> {
    self
      .0
      .get(key)
      .and_then(Value::as_u64)
      .and_then(|value| usize::try_from(value).ok())
  }

  fn text(&self, key: &str) -> Option<String> {
    self.0.get(key).and_then(Value::as_str).map(str::to_owned)
  }
}

/// Reads the events of one training process.
///
/// # Errors
///
/// Returns an error naming the line if a line is not a JSON object, or if a `training` line has no `step`.
pub fn read(ndjson: &str) -> anyhow::Result<TrainingRun> {
  let mut run = TrainingRun::default();
  // A step logged twice in one file keeps the later line.
  let mut steps: BTreeMap<usize, StepLine> = BTreeMap::new();
  for (index, line) in ndjson.lines().enumerate().filter(|(_, line)| !line.trim().is_empty()) {
    let number = index.saturating_add(1);
    let object: Map<String, Value> =
      serde_json::from_str(line).with_context(|| format!("line {number} is not a JSON object"))?;
    let event = Event(&object);
    match object.get("message").and_then(Value::as_str) {
      Some("training") => {
        let Some(step) = event.count("step") else {
          bail!("line {number}: a `training` line has no `step`");
        };
        steps.insert(
          step,
          StepLine {
            step,
            of: event.count("of").unwrap_or(0),
            pair_loss: event.number("pair_loss"),
            group_loss: event.number("group_loss"),
            grad_norm: event.number("grad_norm"),
            rate: event.number("rate"),
          },
        );
      }
      Some("checkpoint") => run.checkpoints.extend(event.count("step")),
      Some("separation") => run.separation.push(SeparationLine {
        split: event.text("split").unwrap_or_default(),
        stage: event.text("stage").unwrap_or_default(),
        step: event.count("step"),
        width: event.count("width").unwrap_or(0),
        edits_mean: event.number("edits_mean"),
        recycles_mean: event.number("recycles_mean"),
        gap_of_means: event.number("gap_of_means"),
        auc: event.number("auc"),
      }),
      Some("finished") => {
        run.finished = Some(Finished {
          out: event.text("out").unwrap_or_default(),
          steps: event.count("steps").unwrap_or(0),
          elapsed_s: event.number("elapsed_s"),
          best_step: event.count("best_step"),
          best_model: event.text("best_model"),
          best_auc: event.number("best_auc"),
        });
      }
      Some("failed" | "panicked") => run.failure = event.text("error"),
      _ => {}
    }
  }
  run.steps = steps.into_values().collect();
  Ok(run)
}

/// Joins the processes of one resumed run, oldest first, into one.
///
/// Each part comes with the step its process went on from (`None` for the first). A process that was resumed
/// from step 120 says nothing new about steps up to 120, and the process before it may have gone further before
/// it stopped: its lines past the resume point are dropped, since those steps were taken again.
pub fn stitch(parts: Vec<(TrainingRun, Option<usize>)>) -> TrainingRun {
  let mut whole = TrainingRun::default();
  for (part, continues_at) in parts {
    if let Some(at) = continues_at {
      whole.steps.retain(|line| line.step <= at);
      whole.checkpoints.retain(|step| *step <= at);
      whole.resumed_at.push(at);
      // The earlier process's closing lines describe a run that was not over.
      whole.finished = None;
      whole.failure = None;
    }
    // A `before` line of a resumed process describes the checkpoint, which an earlier line already did.
    let first = continues_at.is_none();
    whole.steps.extend(part.steps);
    whole.checkpoints.extend(part.checkpoints);
    whole.separation.extend(
      part
        .separation
        .into_iter()
        .filter(|line| first || line.stage != "before"),
    );
    whole.finished = part.finished.or(whole.finished);
    whole.failure = part.failure.or(whole.failure);
  }
  whole
}

impl TrainingRun {
  /// The last logged value of a loss.
  pub fn last(&self, loss: fn(&StepLine) -> Option<f64>) -> Option<f64> {
    self.steps.iter().rev().find_map(loss)
  }

  /// The lowest logged value of a loss, and the step it was logged at.
  pub fn lowest(&self, loss: fn(&StepLine) -> Option<f64>) -> Option<(f64, usize)> {
    self
      .steps
      .iter()
      .filter_map(|line| Some((loss(line)?, line.step)))
      .min_by(|a, b| a.0.total_cmp(&b.0))
  }

  /// The separation line for one split, stage, and width: the last one written, if there are several.
  pub fn separation_at(&self, split: &str, stage: &str, width: usize) -> Option<&SeparationLine> {
    self
      .separation
      .iter()
      .rev()
      .find(|line| line.split == split && line.stage == stage && line.width == width)
  }

  /// Every width some separation line was measured at, narrowest first.
  pub fn widths(&self) -> Vec<usize> {
    let mut widths: Vec<usize> = self.separation.iter().map(|line| line.width).collect();
    widths.sort_unstable();
    widths.dedup();
    widths
  }

  /// The validation AUC of each model the run scored, at `width`, in step order: the base, each checkpoint, and
  /// the final model.
  pub fn validation_curve(&self, width: usize) -> Vec<(usize, f64)> {
    let mut curve: Vec<(usize, f64)> = self
      .separation
      .iter()
      .filter(|line| line.split == "validation" && line.width == width)
      .filter_map(|line| Some((line.step?, line.auc?)))
      .collect();
    curve.sort_by_key(|(step, _)| *step);
    curve.dedup_by_key(|(step, _)| *step);
    curve
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  const FIRST: &str = concat!(
    r#"{"level":"INFO","message":"model loaded","native_width":1024}"#,
    "\n",
    r#"{"message":"separation","split":"validation","stage":"before","step":0,"width":256,"edits_mean":0.9,"recycles_mean":0.6,"gap_of_means":0.3,"auc":0.91,"target":"result"}"#,
    "\n",
    r#"{"message":"training","step":10,"of":40,"epoch":1,"pair_loss":0.05,"grad_norm":1.2,"rate":0.00002}"#,
    "\n",
    r#"{"message":"training","step":20,"of":40,"epoch":2,"pair_loss":0.03,"group_loss":1.5,"grad_norm":1.0,"rate":0.00001}"#,
    "\n",
    r#"{"message":"checkpoint","step":20,"epoch":2,"path":"m.checkpoints/step-000020"}"#,
    "\n",
    r#"{"message":"separation","split":"validation","stage":"checkpoint","step":20,"width":256,"auc":0.95,"target":"result"}"#,
    "\n",
    r#"{"message":"training","step":30,"of":40,"epoch":3,"pair_loss":0.04,"grad_norm":0.9,"rate":0.000005}"#,
    "\n",
    r#"{"message":"failed","error":"out of memory","target":"failure"}"#,
  );

  const SECOND: &str = concat!(
    r#"{"message":"separation","split":"validation","stage":"before","step":20,"width":256,"auc":0.95}"#,
    "\n",
    r#"{"message":"training","step":30,"of":40,"pair_loss":0.025,"grad_norm":0.8,"rate":0.000005}"#,
    "\n",
    r#"{"message":"training","step":40,"of":40,"pair_loss":0.02,"grad_norm":0.7,"rate":0.0}"#,
    "\n",
    r#"{"message":"separation","split":"validation","stage":"after","step":40,"width":256,"auc":0.97}"#,
    "\n",
    r#"{"message":"separation","split":"held_out","stage":"after","step":40,"width":256,"edits_mean":0.95,"recycles_mean":0.5,"gap_of_means":0.45}"#,
    "\n",
    r#"{"message":"finished","out":"models/ft","steps":40,"elapsed_s":61.5,"best_step":40,"best_model":"models/ft","best_auc":0.97,"best_is_final":true}"#,
  );

  #[test]
  fn a_run_is_read_back_from_its_events_by_message_name() {
    let run = read(FIRST).unwrap();
    let steps: Vec<usize> = run.steps.iter().map(|line| line.step).collect();
    assert_eq!(steps, [10, 20, 30]);
    assert_eq!(
      (run.steps[0].of, run.steps[0].pair_loss, run.steps[0].group_loss),
      (40, Some(0.05), None)
    );
    assert_eq!(run.steps[1].group_loss, Some(1.5));
    assert_eq!(run.checkpoints, [20]);
    assert_eq!(run.separation.len(), 2);
    assert_eq!(run.failure.as_deref(), Some("out of memory"));
    assert_eq!(run.finished, None);

    assert_eq!(run.last(|line| line.pair_loss), Some(0.04));
    assert_eq!(run.lowest(|line| line.pair_loss), Some((0.03, 20)));
    assert_eq!(
      run.last(|line| line.group_loss),
      Some(1.5),
      "the last line that has one"
    );
    assert_eq!(run.widths(), [256]);
    assert_eq!(run.validation_curve(256), [(0, 0.91), (20, 0.95)]);
    assert!(run.validation_curve(1024).is_empty());
  }

  #[test]
  fn a_line_that_is_not_json_or_a_training_line_without_a_step_is_refused_by_number() {
    let error = read("{\"message\":\"training\",\"step\":1}\nnot json")
      .unwrap_err()
      .to_string();
    assert!(error.contains("line 2"), "{error}");
    let error = read(r#"{"message":"training","of":4}"#).unwrap_err().to_string();
    assert!(error.contains("no `step`"), "{error}");
    // A step logged twice keeps the later line, and a blank line is nothing.
    let twice = "{\"message\":\"training\",\"step\":1,\"pair_loss\":0.5}\n\n{\"message\":\"training\",\"step\":1,\"pair_loss\":0.4}";
    assert_eq!(read(twice).unwrap().steps[0].pair_loss, Some(0.4));
  }

  /// The first process got to step 30 and died. The second resumed from the checkpoint at 20, so step 30 was
  /// taken twice, and the curve must hold the one that led to the model.
  #[test]
  fn a_resumed_run_is_one_curve_that_follows_the_process_that_went_on() {
    let whole = stitch(vec![(read(FIRST).unwrap(), None), (read(SECOND).unwrap(), Some(20))]);
    let curve: Vec<(usize, Option<f64>)> = whole.steps.iter().map(|line| (line.step, line.pair_loss)).collect();
    assert_eq!(
      curve,
      [(10, Some(0.05)), (20, Some(0.03)), (30, Some(0.025)), (40, Some(0.02))]
    );
    assert_eq!(
      (whole.resumed_at.as_slice(), whole.checkpoints.as_slice()),
      (&[20][..], &[20][..])
    );
    assert_eq!(whole.failure, None, "the run went on and finished");
    let finished = whole.finished.as_ref().unwrap();
    assert_eq!(
      (finished.steps, finished.best_step, finished.best_auc),
      (40, Some(40), Some(0.97))
    );

    // The base is scored once, by the process that started from it.
    assert_eq!(whole.validation_curve(256), [(0, 0.91), (20, 0.95), (40, 0.97)]);
    let held = whole.separation_at("held_out", "after", 256).unwrap();
    assert_eq!(held.gap_of_means, Some(0.45));
    assert!(whole.separation_at("held_out", "before", 256).is_none());

    // One process alone is itself.
    assert_eq!(stitch(vec![(read(FIRST).unwrap(), None)]), read(FIRST).unwrap());
  }
}
