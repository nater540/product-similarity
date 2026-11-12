//! Replaying whole edit histories, to measure drift.
//!
//! The drift check (`anchor_threshold`) rejects a revision that is too far from the record as first stored,
//! however small each step was. Where that threshold belongs cannot be read from before/after pairs: it is a
//! question about a product's whole life. This replays each history through the real engine, version by version,
//! and keeps what every step measured: against the version before it, and against the anchor.
//!
//! **Measured first, judged afterwards.** Replaying under the configured policy would be wrong: after the first
//! rejected version the stored record no longer moves, so every later version would be compared with a baseline
//! the real product left long ago. The replay therefore runs on an engine that can reject nothing (the same
//! fields and weights, every limit cleared), so that every version is written and the anchor follows the engine's
//! own rules. Each step's measurements are then judged under the configured schema and policy
//! (`Evaluation::rejudge`). Up to and including the first rejection that is exactly what the server would have
//! decided, and a test holds the two together.

use anyhow::{Context, bail};
use similarity_core::score::{self, Comparison, Evaluation};
use similarity_core::{FieldSpec, Policy, Quantity, QuantityScore, QuantitySpec, Schema, quantity};
use similarity_store::{MemoryStore, VectorStore};

use super::RecordFields;
use super::histories::{Histories, History};
use super::pairs::Input;
use crate::engine::{
  Engine, EvaluateInput, ImportInput, Imported, Original, RecordInput, Verdict, WriteMode, store_layout,
};
use crate::worker::TextEmbedder;

/// One step of a history: a version judged as a revision of the version before it.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
  /// The index of the version in the history, from 1.
  pub step: usize,
  /// When the version was saved, as the file wrote it.
  pub at: Option<String>,
  /// What the step measured, judged under the configured schema and policy.
  pub evaluation: Evaluation,
  /// Whether the same measurements pass with the drift check taken away: the rules a pair is judged by.
  pub accepted_by_pair_rules: bool,
  /// Each field's similarity to the anchor, `None` where the version or the anchor has no value for it.
  pub anchor_fields: Vec<(String, Option<f32>)>,
  /// Each quantity compared with the first value the record had for it.
  pub anchor_quantities: Vec<QuantityScore>,
  /// Whether the version changed anything. One that is identical to the version before it is not written.
  pub written: bool,
}

impl Step {
  /// Whether the step's similarity to the anchor is below `threshold`.
  pub fn drifted_below(&self, threshold: f32) -> bool {
    self
      .evaluation
      .anchor_similarity
      .is_some_and(|similarity| similarity < threshold)
  }
}

/// A history and what each of its steps measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Replayed {
  /// The history.
  pub history: History,
  /// One step per version after the first.
  pub steps: Vec<Step>,
}

/// Every history of a file, replayed.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftRun {
  /// The histories, in the file's order.
  pub histories: Vec<Replayed>,
  /// The file's fingerprint, the selection, and the record keys that were not compared.
  pub input: Input,
}

/// `schema` with every limit cleared: the same fields, weights, and quantities, so the same vectors and the same
/// measurements, and nothing that can reject.
fn measuring(schema: &Schema) -> anyhow::Result<Schema> {
  let fields = schema
    .fields()
    .iter()
    .map(|field| FieldSpec {
      reject_below: None,
      ..field.clone()
    })
    .collect();
  let quantities = schema
    .quantities()
    .iter()
    .map(|spec| QuantitySpec {
      name: spec.name.clone(),
      reject_above_factor: None,
      reject_incomparable: false,
    })
    .collect();
  Ok(Schema::new(fields)?.with_quantities(quantities)?)
}

fn texts(record: &RecordFields) -> Vec<(String, String)> {
  record
    .texts
    .iter()
    .map(|(name, text)| (name.clone(), text.clone()))
    .collect()
}

fn quantities(record: &RecordFields) -> Vec<(String, Quantity)> {
  record
    .quantities
    .iter()
    .map(|(name, quantity)| (name.clone(), quantity.clone()))
    .collect()
}

fn record_input(history: &History, record: &RecordFields) -> RecordInput {
  RecordInput {
    id: Some(history.id.clone()),
    brand_id: history.brand_id.clone(),
    fields: texts(record),
    quantities: quantities(record),
  }
}

/// Stores a history's first version, with the product as created as its anchor where the file says what that was.
async fn seed<S: VectorStore, E: TextEmbedder>(engine: &Engine<S, E>, history: &History) -> anyhow::Result<()> {
  let Some(first) = history.versions.first() else {
    bail!("`{}` has no versions", history.id);
  };
  let seed = ImportInput {
    record: record_input(history, &first.fields),
    anchor: history.original.as_ref().map(|original| Original {
      fields: texts(original),
      quantities: quantities(original),
    }),
  };
  let imported = engine
    .import(vec![seed])
    .await
    .with_context(|| format!("`{}`, version 0", history.id))?;
  match imported.into_iter().next() {
    Some(Ok(Imported::Created)) => Ok(()),
    Some(Ok(Imported::Skipped)) => bail!("`{}` is in the file twice", history.id),
    Some(Err(error)) => Err(error).with_context(|| format!("`{}`, version 0", history.id)),
    None => bail!("`{}` was not stored", history.id),
  }
}

/// Replays every history through an engine of its own, on a store in memory, and judges each step under `schema`
/// and `policy`.
///
/// # Errors
///
/// Returns an error naming the product and the version if a version does not fit the schema or the model fails.
pub async fn replay<E: TextEmbedder>(
  embedder: E,
  schema: &Schema,
  policy: &Policy,
  histories: Histories,
) -> anyhow::Result<DriftRun> {
  let dimension = embedder.dimension();
  let measuring_schema = measuring(schema)?;
  let (layout, composite) = store_layout(&measuring_schema, &embedder);
  // A threshold of -1 is below every cosine there is, and nothing else is set: no version can be refused.
  let engine = Engine::new(
    MemoryStore::new(layout, composite),
    embedder,
    measuring_schema,
    Policy::new(-1.0, None)?,
  );
  let pair_rules = Policy {
    anchor_threshold: None,
    ..*policy
  };

  let mut replayed = Vec::with_capacity(histories.histories.len());
  for history in histories.histories {
    seed(&engine, &history).await?;

    let mut steps = Vec::with_capacity(history.versions.len().saturating_sub(1));
    for (index, version) in history.versions.iter().enumerate().skip(1) {
      let place = || format!("`{}`, version {index}", history.id);
      // The anchor this step is held to is the one stored before it.
      let stored = engine
        .store()
        .get(&history.id)
        .await
        .with_context(place)?
        .with_context(|| format!("{}: the record is gone from the replay's store", place()))?;
      let output = engine
        .evaluate(EvaluateInput {
          record: record_input(&history, &version.fields),
          mode: WriteMode::Enforce,
          threshold: None,
        })
        .await
        .with_context(place)?;
      if output.verdict == Verdict::Rejected {
        bail!("{}: the measuring engine refused a version, which it cannot", place());
      }
      let mut evaluation = Evaluation {
        comparison: Comparison {
          similarity: output.similarity.unwrap_or(1.0),
          fields: output.fields,
        },
        anchor_similarity: output.anchor_similarity,
        reasons: Vec::new(),
        quantities: output.quantities,
        measure_changed: output.measure_changed,
      };
      let mut by_pair_rules = evaluation.clone();
      evaluation.rejudge(schema, policy);
      by_pair_rules.rejudge(schema, &pair_rules);
      let anchor_quantities = schema
        .quantities()
        .iter()
        .filter_map(|spec| {
          quantity::against_first(
            &spec.name,
            stored.anchor_quantities.get(&spec.name),
            version.fields.quantities.get(&spec.name),
          )
        })
        .collect();
      steps.push(Step {
        step: index,
        at: version.at.clone(),
        accepted_by_pair_rules: by_pair_rules.accepted(),
        anchor_fields: score::field_similarities(schema, dimension, &stored.anchor, &output.vector)
          .with_context(place)?,
        anchor_quantities,
        written: output.stored,
        evaluation,
      });
    }
    replayed.push(Replayed { history, steps });
  }
  Ok(DriftRun {
    histories: replayed,
    input: histories.input,
  })
}

#[cfg(test)]
pub(crate) mod testing {
  //! Histories for tests, on the word-bag embedder.

  use similarity_core::{FieldSpec, Policy, QuantitySpec, Schema};

  use super::{DriftRun, replay};
  use crate::offline::histories::parse_histories;
  use crate::worker::testing::WordBagEmbedder;

  /// Two text fields and a weight. The name has a floor and the weight a limit, so every kind of reason can occur.
  pub(crate) fn schema() -> Schema {
    let field = |name: &str, weight: f32, reject_below: Option<f32>| FieldSpec {
      name: name.to_owned(),
      weight,
      required: false,
      reject_below,
    };
    Schema::new(vec![field("name", 2.0, Some(0.3)), field("description", 1.0, None)])
      .unwrap()
      .with_quantities(vec![QuantitySpec {
        name: "weight".to_owned(),
        reject_above_factor: Some(4.0),
        reject_incomparable: false,
      }])
      .unwrap()
  }

  pub(crate) fn policy() -> Policy {
    Policy::new(0.6, Some(0.5)).unwrap()
  }

  /// One history line: a label, then each version as `(name, description, weight in grams)`.
  pub(crate) fn line(product: &str, rest: &str, versions: &[(&str, Option<&str>, Option<f64>)]) -> String {
    let versions: Vec<String> = versions
      .iter()
      .map(|(name, description, weight)| {
        let description = description.map_or_else(String::new, |text| format!(r#", "description": "{text}""#));
        let weight = weight.map_or_else(String::new, |grams| {
          format!(r#", "weight": {{"value": {grams}, "unit": "g"}}"#)
        });
        format!(r#"{{"fields": {{"name": "{name}"{description}{weight}}}}}"#)
      })
      .collect();
    format!(
      r#"{{"product": "{product}", {rest}"versions": [{}]}}"#,
      versions.join(", ")
    )
  }

  pub(crate) fn replayed(lines: &[String]) -> DriftRun {
    let histories = parse_histories(&lines.join("\n"), &schema(), None).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    runtime
      .block_on(replay(
        WordBagEmbedder::new("word-bag-v1"),
        &schema(),
        &policy(),
        histories,
      ))
      .unwrap()
  }
}

#[cfg(test)]
mod tests {
  use similarity_core::{QuantityChange, Reason};

  use super::testing::{line, policy, replayed, schema};
  use super::*;
  use crate::offline::histories::parse_histories;
  use crate::worker::testing::WordBagEmbedder;

  /// A slow walk from gummies to a vape cartridge, one word at a time, with a legitimate first edit.
  const WALK: [(&str, Option<&str>, Option<f64>); 6] = [
    ("blueberry gummies edibles sweet", None, Some(40.0)),
    ("blueberry gummies edibles sweet vegan", None, Some(40.0)),
    ("blueberry gummies edibles cartridge vegan", None, Some(40.0)),
    ("blueberry vape edibles cartridge vegan", None, Some(40.0)),
    ("sativa vape resin cartridge vegan", None, Some(40.0)),
    ("sativa vape resin cartridge distillate", None, Some(40.0)),
  ];

  #[test]
  fn every_version_is_measured_against_the_one_before_it_and_against_the_anchor() {
    let run = replayed(&[line(
      "walk",
      r#""brand_id": "b1", "label": "recycle", "recycled_at": 2, "#,
      &WALK,
    )]);
    let [history] = run.histories.as_slice() else {
      panic!("one history")
    };
    assert_eq!(history.steps.len(), 5, "one step per version after the first");
    assert!(history.steps.iter().all(|step| step.written));

    // Each step is small, and the distance from the first version only grows.
    let to_anchor: Vec<f32> = history
      .steps
      .iter()
      .map(|step| step.evaluation.anchor_similarity.unwrap())
      .collect();
    assert!(
      to_anchor.windows(2).all(|pair| pair[1] <= pair[0] + 1e-6),
      "{to_anchor:?}"
    );
    assert!(to_anchor[0] > 0.8 && to_anchor[4] < 0.3, "{to_anchor:?}");
    let step_by_step: Vec<f32> = history
      .steps
      .iter()
      .map(|step| step.evaluation.comparison.similarity)
      .collect();
    assert!(
      step_by_step.iter().all(|similarity| *similarity > 0.55),
      "{step_by_step:?}"
    );

    // The anchor is the first version, and the name is what moved.
    let last = history.steps.last().unwrap();
    let name = last.anchor_fields.iter().find(|(field, _)| field == "name").unwrap();
    assert!(
      (name.1.unwrap() - to_anchor[4]).abs() < 1e-5,
      "only the name has a value on both sides"
    );
    assert_eq!(
      last
        .anchor_fields
        .iter()
        .find(|(field, _)| field == "description")
        .unwrap()
        .1,
      None
    );
    assert_eq!(last.anchor_quantities[0].change, QuantityChange::Unchanged);
  }

  #[test]
  fn a_slow_walk_passes_the_pair_rules_and_is_stopped_by_the_drift_check() {
    let run = replayed(&[line("walk", r#""label": "recycle", "recycled_at": 2, "#, &WALK)]);
    let steps = &run.histories[0].steps;
    let drifted = steps
      .iter()
      .find(|step| !step.evaluation.accepted())
      .expect("the walk is stopped somewhere");
    assert!(drifted.accepted_by_pair_rules, "the step itself is small");
    assert!(
      matches!(&drifted.evaluation.reasons[..], [Reason::AnchorDrift { .. }]),
      "{:?}",
      drifted.evaluation.reasons
    );
    assert!(drifted.drifted_below(0.5) && !steps[0].drifted_below(0.5));
    assert!(steps[0].evaluation.accepted());
  }

  #[test]
  fn a_quantity_walked_up_in_small_steps_is_far_from_its_first_value() {
    let name = "indica flower";
    let versions = [
      (name, None, Some(3.5)),
      (name, None, Some(7.0)),
      (name, None, Some(14.0)),
      (name, None, Some(28.0)),
      (name, None, None),
      (name, None, Some(56.0)),
    ];
    let run = replayed(&[line("weights", r#""label": "recycle", "#, &versions)]);
    let steps = &run.histories[0].steps;
    let step_factors: Vec<Option<f64>> = steps.iter().map(|step| step.evaluation.quantities[0].factor).collect();
    assert_eq!(
      &step_factors[..3],
      [Some(2.0), Some(2.0), Some(2.0)],
      "each doubling is within the limit of four"
    );
    assert!(steps[..3].iter().all(|step| step.evaluation.accepted()));
    let first_factors: Vec<Option<f64>> = steps.iter().map(|step| step.anchor_quantities[0].factor).collect();
    assert_eq!(&first_factors[..3], [Some(2.0), Some(4.0), Some(8.0)]);

    // Removed, then back at sixteen times the first value: the refill is held to the anchor, by the engine's rule.
    assert_eq!(steps[3].anchor_quantities[0].change, QuantityChange::Removed);
    let refilled = &steps[4];
    assert_eq!(refilled.evaluation.quantities[0].change, QuantityChange::Restored);
    assert_eq!(refilled.anchor_quantities[0].factor, Some(16.0));
    assert!(matches!(
      &refilled.evaluation.reasons[..],
      [Reason::QuantityBeyondFactor { .. }]
    ));
  }

  #[test]
  fn a_history_with_its_original_is_measured_from_the_original() {
    let versions = [
      ("sativa vape resin cartridge", None, None),
      ("sativa vape resin cartridge pen", None, None),
    ];
    let seeded = line(
      "seeded",
      r#""label": "recycle", "original": {"name": "blueberry gummies edibles sweet", "description": "ten gummies"}, "#,
      &versions,
    );
    let unseeded = line("unseeded", r#""label": "edit", "truncated": true, "#, &versions);
    let run = replayed(&[seeded, unseeded]);
    let (seeded, unseeded) = (&run.histories[0].steps[0], &run.histories[1].steps[0]);
    assert!(
      seeded.evaluation.anchor_similarity.unwrap() < 0.2,
      "far from what it was created as"
    );
    assert!(
      unseeded.evaluation.anchor_similarity.unwrap() > 0.8,
      "measured from where the file starts"
    );
    assert!((seeded.evaluation.comparison.similarity - unseeded.evaluation.comparison.similarity).abs() < 1e-6);
  }

  /// Measuring everything and judging afterwards must agree with the server up to the first rejection: that is
  /// the whole claim the replay rests on.
  #[test]
  fn judging_afterwards_gives_the_first_rejection_the_server_would_have_given() {
    let lines = [
      line("walk", r#""label": "recycle", "#, &WALK),
      line(
        "swap",
        r#""label": "recycle", "#,
        &[
          ("blueberry gummies", Some("ten vegan gummies"), Some(40.0)),
          ("blueberry gummies", None, Some(40.0)),
          ("blueberry gummies", Some("one gram cartridge"), Some(40.0)),
        ],
      ),
      line(
        "jump",
        r#""label": "recycle", "#,
        &[
          ("blueberry gummies", None, Some(40.0)),
          ("sativa vape cartridge", None, Some(1.0)),
          ("sativa vape", None, Some(1.0)),
        ],
      ),
      line(
        "steady",
        r#""label": "edit", "#,
        &[
          ("indica flower", None, Some(3.5)),
          ("indica flower premium", None, Some(3.5)),
        ],
      ),
    ];
    let run = replayed(&lines);
    let histories = parse_histories(&lines.join("\n"), &schema(), None).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    for (replayed, history) in run.histories.iter().zip(&histories.histories) {
      let expected = replayed
        .steps
        .iter()
        .find(|step| !step.evaluation.accepted())
        .map(|step| step.step);
      // The server itself, under the configured policy, in the order the versions arrived.
      let embedder = WordBagEmbedder::new("word-bag-v1");
      let (layout, dimension) = store_layout(&schema(), &embedder);
      let engine = Engine::new(MemoryStore::new(layout, dimension), embedder, schema(), policy());
      let first_rejected = runtime.block_on(async {
        for (index, version) in history.versions.iter().enumerate() {
          let output = engine
            .evaluate(EvaluateInput {
              record: RecordInput {
                id: Some(history.id.clone()),
                brand_id: None,
                fields: texts(&version.fields),
                quantities: quantities(&version.fields),
              },
              mode: WriteMode::Enforce,
              threshold: None,
            })
            .await
            .unwrap();
          if output.verdict == Verdict::Rejected {
            let codes = |reasons: &[Reason]| reasons.iter().map(Reason::code).collect::<Vec<_>>();
            let step = replayed.steps.iter().find(|step| step.step == index).unwrap();
            assert_eq!(
              codes(&output.reasons),
              codes(&step.evaluation.reasons),
              "{}",
              history.id
            );
            return Some(index);
          }
        }
        None
      });
      assert_eq!(first_rejected, expected, "{}", history.id);
    }
    let rejected: Vec<bool> = run
      .histories
      .iter()
      .map(|history| history.steps.iter().any(|step| !step.evaluation.accepted()))
      .collect();
    assert_eq!(
      rejected,
      [true, false, true, false],
      "the cases cover a walk, a jump, and two that pass"
    );
  }
}
