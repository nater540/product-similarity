//! Properties of the service, checked on generated inputs.
//!
//! The engine's tests are scenarios someone thought of. This drives the engine with random sequences of requests
//! (every write mode, with and without quantities and tenants, valid and not) and after every single request
//! checks the rules the product stands on, whatever came before:
//!
//! - a rejected revision, a dry run, and a failed request leave the stored record exactly as it was (D7);
//! - an accepted write never replaces a value the anchor holds, text or quantity, and gives it only the first
//!   value of something it never had (D8, D16);
//! - a forced write makes the record its own anchor;
//! - the revision counter moves by exactly one, exactly when something was stored;
//! - a record with a tenant is touched only by a request that names it (D14);
//! - nothing ever happens to a record other than the one a request is about.
//!
//! It runs on the in-memory store and the bag-of-words embedder, so it needs no model and no Qdrant. See
//! `similarity-core`'s `properties` module for the scoring properties and for how failures are saved.

use proptest::prelude::*;
use proptest::test_runner::Config;
use similarity_core::score::Evaluation;
use similarity_core::vector::cosine;
use similarity_core::{FieldSpec, Policy, Quantity, QuantitySpec, Schema};
use similarity_store::{MemoryStore, StoredRecord, VectorStore};

use crate::engine::{
  Engine, EngineError, EvaluateInput, FindSimilarInput, RecordInput, SimilarTo, Verdict, WriteMode, store_layout,
};
use crate::offline::calibrate::suggest_threshold;
use crate::offline::pairs::Label;
use crate::offline::{RecordFields, Suggestion};
use crate::worker::testing::{WORD_BAG_DIMENSION, WordBagEmbedder};

fn cases() -> Config {
  let cases = std::env::var("PROPTEST_CASES")
    .ok()
    .and_then(|cases| cases.parse().ok())
    .unwrap_or(64);
  Config {
    cases,
    ..Config::default()
  }
}

/// The ids requests are about. Few, so that a sequence keeps coming back to the same records.
const IDS: [&str; 2] = ["a", "b"];
const TENANTS: [&str; 2] = ["brand-1", "brand-2"];
/// Names that share words in a chain, so that some edits are small, some large, and some walk a record away.
const NAMES: [&str; 7] = [
  "blueberry gummies vegan fruit chews",
  "blueberry gummies vegan fruit cartridge",
  "blueberry gummies vegan vape cartridge",
  "blueberry gummies resin vape cartridge",
  "sour diesel resin vape cartridge",
  "blueberry gummies",
  "   ",
];
const CATEGORIES: [&str; 4] = ["edibles gummies", "vape pens", "edibles", ""];
const DESCRIPTIONS: [&str; 3] = ["ten vegan gummies with real fruit", "one gram of live resin", "  "];

/// One request, as indexes into the lists above.
#[derive(Debug, Clone)]
enum Request {
  Evaluate {
    id: Option<usize>,
    tenant: Option<usize>,
    name: usize,
    category: Option<usize>,
    description: Option<usize>,
    weight: Option<(f64, &'static str)>,
    thc: Option<(f64, &'static str)>,
    mode: WriteMode,
    threshold: Option<f32>,
  },
  Delete {
    id: usize,
    tenant: Option<usize>,
  },
  Search {
    id: usize,
  },
}

fn quantity() -> impl Strategy<Value = (f64, &'static str)> {
  (
    prop_oneof![Just(0.0), Just(1.0), Just(3.5), Just(28.0), Just(1000.0), Just(-1.0)],
    prop_oneof![Just("g"), Just("mg"), Just("%"), Just("each"), Just(" ")],
  )
}

fn request() -> impl Strategy<Value = Request> {
  let evaluate = (
    (
      proptest::option::weighted(0.9, 0..IDS.len()),
      proptest::option::of(0..TENANTS.len()),
      // Mostly real names; the blank one makes the request invalid.
      prop_oneof![9 => 0..NAMES.len() - 1, 1 => Just(NAMES.len() - 1)],
      proptest::option::of(0..CATEGORIES.len()),
      proptest::option::of(0..DESCRIPTIONS.len()),
    ),
    (
      proptest::option::of(quantity()),
      proptest::option::of(quantity()),
      prop_oneof![
        6 => Just(WriteMode::Enforce),
        2 => Just(WriteMode::DryRun),
        1 => Just(WriteMode::Force),
      ],
      prop_oneof![8 => Just(None), 1 => Just(Some(0.2_f32)), 1 => Just(Some(7.0_f32))],
    ),
  )
    .prop_map(
      |((id, tenant, name, category, description), (weight, thc, mode, threshold))| Request::Evaluate {
        id,
        tenant,
        name,
        category,
        description,
        weight,
        thc,
        mode,
        threshold,
      },
    );
  let delete =
    (0..IDS.len(), proptest::option::of(0..TENANTS.len())).prop_map(|(id, tenant)| Request::Delete { id, tenant });
  let search = (0..IDS.len()).prop_map(|id| Request::Search { id });
  prop_oneof![12 => evaluate, 2 => delete, 1 => search]
}

fn schema() -> Schema {
  let field = |name: &str, weight: f32, required: bool, reject_below: Option<f32>| FieldSpec {
    name: name.to_owned(),
    weight,
    required,
    reject_below,
  };
  let quantity = |name: &str, reject_above_factor: Option<f64>, reject_incomparable: bool| QuantitySpec {
    name: name.to_owned(),
    reject_above_factor,
    reject_incomparable,
  };
  Schema::new(vec![
    field("name", 4.0, true, Some(0.3)),
    field("category", 2.0, false, None),
    field("description", 1.0, false, None),
  ])
  .unwrap()
  .with_quantities(vec![quantity("weight", Some(4.0), true), quantity("thc", None, false)])
  .unwrap()
}

type TestEngine = Engine<MemoryStore, WordBagEmbedder>;

fn engine(anchor_threshold: Option<f32>, reject_on_measure_change: bool) -> TestEngine {
  let embedder = WordBagEmbedder::new("word-bag-v1");
  let (layout, dimension) = store_layout(&schema(), &embedder);
  let mut policy = Policy::new(0.75, anchor_threshold).unwrap();
  policy.reject_on_measure_change = reject_on_measure_change;
  Engine::new(MemoryStore::new(layout, dimension), embedder, schema(), policy)
}

async fn stored(engine: &TestEngine) -> Vec<Option<StoredRecord>> {
  let mut records = Vec::new();
  for id in IDS {
    records.push(engine.store().get(id).await.unwrap());
  }
  records
}

/// Whether the request is one the engine must refuse as malformed, whatever is stored.
fn is_invalid(name: usize, weight: Option<(f64, &str)>, thc: Option<(f64, &str)>, threshold: Option<f32>) -> bool {
  let bad = |quantity: Option<(f64, &str)>| quantity.is_some_and(|(value, unit)| value < 0.0 || unit.trim().is_empty());
  NAMES[name].trim().is_empty()
    || bad(weight)
    || bad(thc)
    || threshold.is_some_and(|threshold| !(-1.0..=1.0).contains(&threshold))
}

/// A failure of the code under test that is not the rule being checked: the case fails with what went wrong.
fn sound<T>(result: similarity_core::Result<T>) -> Result<T, TestCaseError> {
  result.map_err(|error| TestCaseError::fail(error.to_string()))
}

/// D8 and D16, for one accepted ordinary write: everything the anchor held is still there, unchanged, and what it
/// gained is the candidate's.
fn check_anchor_kept(before: &StoredRecord, after: &StoredRecord) -> Result<(), TestCaseError> {
  let schema = schema();
  let old = sound(schema.blocks(WORD_BAG_DIMENSION, &before.anchor))?;
  let new = sound(schema.blocks(WORD_BAG_DIMENSION, &after.anchor))?;
  let current = sound(schema.blocks(WORD_BAG_DIMENSION, &after.vectors.composite))?;
  for (((held, kept), candidate), spec) in old.iter().zip(&new).zip(&current).zip(schema.fields()) {
    match (held, kept) {
      (Some(held), Some(kept)) => {
        let agreement = sound(cosine(held, kept))?;
        prop_assert!(agreement > 0.9999, "the anchor's `{}` moved: {}", spec.name, agreement);
      }
      (Some(_), None) => prop_assert!(false, "the anchor lost `{}`", spec.name),
      (None, Some(kept)) => {
        let Some(candidate) = candidate else {
          return Err(TestCaseError::fail(format!(
            "the anchor gained `{}` from nowhere",
            spec.name
          )));
        };
        prop_assert!(sound(cosine(candidate, kept))? > 0.9999);
      }
      (None, None) => prop_assert!(candidate.is_none(), "the anchor did not take the new `{}`", spec.name),
    }
  }
  for (name, held) in &before.anchor_quantities {
    prop_assert_eq!(
      after.anchor_quantities.get(name),
      Some(held),
      "the anchor's `{}` was replaced",
      name
    );
  }
  for (name, kept) in &after.anchor_quantities {
    if !before.anchor_quantities.contains_key(name) {
      prop_assert_eq!(
        after.quantities.get(name),
        Some(kept),
        "the anchor gained `{}` from nowhere",
        name
      );
    }
  }
  Ok(())
}

/// Runs one request against the engine and checks every rule that applies to it.
#[expect(
  clippy::too_many_lines,
  reason = "one request, and every rule it must obey, read in one place"
)]
async fn step(engine: &TestEngine, request: &Request) -> Result<(), TestCaseError> {
  let before = stored(engine).await;
  match request.clone() {
    Request::Evaluate {
      id,
      tenant,
      name,
      category,
      description,
      weight,
      thc,
      mode,
      threshold,
    } => {
      let mut fields = vec![("name".to_owned(), NAMES[name].to_owned())];
      fields.extend(category.map(|index| ("category".to_owned(), CATEGORIES[index].to_owned())));
      fields.extend(description.map(|index| ("description".to_owned(), DESCRIPTIONS[index].to_owned())));
      let mut quantities = Vec::new();
      quantities.extend(weight.map(|(value, unit)| ("weight".to_owned(), Quantity::single(value, unit))));
      quantities.extend(thc.map(|(value, unit)| ("thc".to_owned(), Quantity::single(value, unit))));
      let tenant = tenant.map(|index| TENANTS[index].to_owned());
      let input = EvaluateInput {
        record: RecordInput {
          id: id.map(|index| IDS[index].to_owned()),
          tenant: tenant.clone(),
          fields,
          quantities,
        },
        mode,
        threshold,
      };
      let outcome = engine.evaluate(input).await;
      let after = stored(engine).await;

      // Whatever happened, it happened to at most the record the request named.
      for (index, (was, is)) in before.iter().zip(&after).enumerate() {
        if Some(index) != id {
          prop_assert_eq!(
            was,
            is,
            "a request about {:?} changed `{}`",
            id.map(|id| IDS[id]),
            IDS[index]
          );
        }
      }
      let (was, is) = match id {
        Some(index) => (before[index].as_ref(), after[index].as_ref()),
        None => (None, None),
      };

      let owner = was.and_then(|record| record.tenant.as_ref());
      let wrong_tenant = owner.is_some_and(|owner| Some(owner) != tenant.as_ref());
      let output = match outcome {
        Err(error) => {
          prop_assert_eq!(was, is, "a failed request wrote something: {:?}", error);
          if is_invalid(name, weight, thc, threshold) {
            prop_assert!(matches!(error, EngineError::InvalidArgument(_)), "{:?}", error);
          } else {
            prop_assert!(wrong_tenant, "an acceptable request was refused: {:?}", error);
            prop_assert!(matches!(error, EngineError::FailedPrecondition(_)), "{:?}", error);
          }
          return Ok(());
        }
        Ok(output) => output,
      };
      prop_assert!(
        !is_invalid(name, weight, thc, threshold),
        "a malformed request was served"
      );
      prop_assert!(
        !wrong_tenant,
        "D14: a record with a tenant was evaluated by a request that does not name it"
      );
      if let Some(similarity) = output.similarity {
        prop_assert!(similarity.is_finite() && (-1.0..=1.0).contains(&similarity));
      }

      // The verdict follows from whether there was something to compare with.
      let expected = match (id, was) {
        (None, _) => Some(Verdict::Embedded),
        (Some(_), None) => Some(Verdict::Created),
        (Some(_), Some(_)) => None,
      };
      if let Some(expected) = expected {
        prop_assert_eq!(output.verdict, expected);
      } else {
        prop_assert!(matches!(output.verdict, Verdict::Accepted | Verdict::Rejected));
        prop_assert_eq!(output.verdict == Verdict::Rejected, !output.reasons.is_empty());
      }

      // D7: what may be written, and what was.
      match (mode, output.verdict) {
        (_, Verdict::Embedded) | (WriteMode::DryRun, _) | (WriteMode::Enforce, Verdict::Rejected) => {
          prop_assert!(!output.stored, "{:?} in {:?} must not write", output.verdict, mode);
        }
        (WriteMode::Force, _) | (WriteMode::Enforce, Verdict::Created) => {
          prop_assert!(output.stored, "{:?} in {:?} must write", output.verdict, mode);
        }
        (WriteMode::Enforce, Verdict::Accepted) => {}
      }
      if !output.stored {
        prop_assert_eq!(was, is, "D7: nothing was stored, yet the record changed");
        prop_assert_eq!(output.revision, was.map_or(0, |record| record.revision));
        return Ok(());
      }

      let Some(is) = is else {
        return Err(TestCaseError::fail("the response says stored, and nothing is"));
      };
      prop_assert_eq!(
        is.revision,
        was.map_or(0, |record| record.revision) + 1,
        "one write, one revision"
      );
      prop_assert_eq!(output.revision, is.revision);
      prop_assert_eq!(&is.tenant, &tenant);
      match (was, mode) {
        // A new record and a forced one are their own point of reference.
        (None, _) | (Some(_), WriteMode::Force) => {
          prop_assert_eq!(&is.anchor, &is.vectors.composite);
          prop_assert_eq!(&is.anchor_quantities, &is.quantities);
        }
        (Some(was), _) => check_anchor_kept(was, is)?,
      }
    }
    Request::Delete { id, tenant } => {
      let tenant = tenant.map(|index| TENANTS[index].to_owned());
      let outcome = engine.delete(IDS[id].to_owned(), tenant.clone()).await;
      let after = stored(engine).await;
      for (index, (was, is)) in before.iter().zip(&after).enumerate() {
        if index != id {
          prop_assert_eq!(was, is);
        }
      }
      let owner = before[id].as_ref().and_then(|record| record.tenant.as_ref());
      if owner.is_some_and(|owner| Some(owner) != tenant.as_ref()) {
        prop_assert!(
          matches!(outcome, Err(EngineError::FailedPrecondition(_))),
          "D14: {:?}",
          outcome
        );
        prop_assert_eq!(&before[id], &after[id]);
      } else {
        prop_assert_eq!(outcome, Ok(before[id].is_some()));
        prop_assert!(after[id].is_none());
      }
    }
    Request::Search { id } => {
      let outcome = engine
        .find_similar(FindSimilarInput {
          query: SimilarTo::Id(IDS[id].to_owned()),
          limit: 0,
          tenant: None,
          min_similarity: None,
        })
        .await;
      prop_assert_eq!(&before, &stored(engine).await, "a search wrote something");
      match (&before[id], outcome) {
        (None, outcome) => prop_assert!(matches!(outcome, Err(EngineError::NotFound(_))), "{:?}", outcome),
        (Some(_), outcome) => {
          let neighbors = outcome.unwrap();
          let others = before.iter().flatten().count() - 1;
          prop_assert_eq!(
            neighbors.len(),
            others,
            "every other stored record, and never the record itself"
          );
          prop_assert!(neighbors.iter().all(|neighbor| neighbor.id != IDS[id]));
        }
      }
    }
  }
  Ok(())
}

proptest! {
  #![proptest_config(cases())]

  /// Any sequence of requests, under any of the policies, obeys every rule at every step.
  #[test]
  fn every_request_in_any_sequence_obeys_the_write_rules(
    requests in proptest::collection::vec(request(), 1..40),
    anchor_threshold in proptest::option::of(Just(0.5_f32)),
    reject_on_measure_change in any::<bool>(),
  ) {
    let engine = engine(anchor_threshold, reject_on_measure_change);
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    runtime.block_on(async {
      for (index, request) in requests.iter().enumerate() {
        step(&engine, request)
          .await
          .map_err(|error| TestCaseError::fail(format!("request {index}, {request:?}: {error}")))?;
      }
      Ok::<(), TestCaseError>(())
    })?;
  }

  /// The suggested threshold is one no other cut between neighboring scores beats: fewest pairs misclassified,
  /// and among those the widest margin.
  #[test]
  fn the_suggested_threshold_is_what_trying_every_cut_finds(
    scores in proptest::collection::vec((any::<bool>(), 0_u16..=1000), 0..24),
  ) {
    let scores: Vec<(Label, f32)> = scores
      .into_iter()
      .map(|(is_edit, score)| (if is_edit { Label::Edit } else { Label::Recycle }, f32::from(score) / 1000.0))
      .collect();
    let errors_at = |threshold: f32| {
      scores
        .iter()
        .filter(|(label, score)| (*label == Label::Edit) != (*score >= threshold))
        .count()
    };
    let has = |label: Label| scores.iter().any(|(found, _)| *found == label);
    let mut distinct: Vec<f32> = scores.iter().map(|(_, score)| *score).collect();
    distinct.sort_by(f32::total_cmp);
    distinct.dedup();

    let suggestion = suggest_threshold(&scores);
    if !(has(Label::Edit) && has(Label::Recycle)) || distinct.len() < 2 {
      prop_assert_eq!(suggestion, None);
      return Ok(());
    }
    let Suggestion { threshold, margin, false_rejects, false_accepts } = suggestion.expect("both labels, two scores");
    prop_assert_eq!(false_rejects + false_accepts, errors_at(threshold), "its counts are its own errors");

    let cuts = distinct.windows(2).map(|pair| (f32::midpoint(pair[0], pair[1]), (pair[1] - pair[0]) / 2.0));
    for (cut, cut_margin) in cuts {
      let better = errors_at(cut) < errors_at(threshold)
        || (errors_at(cut) == errors_at(threshold) && cut_margin > margin + 1e-6);
      prop_assert!(!better, "the cut at {} beats the suggested {}", cut, threshold);
    }
  }

  /// A record file is one JSON object; written and read back it is the same record.
  #[test]
  fn a_record_written_as_json_reads_back_the_same(
    texts in proptest::collection::btree_map("[a-c]{1,3}", "[a-zA-Z0-9 >%é]{0,12}", 0..4),
    quantities in proptest::collection::btree_map("[d-f]{1,3}", (0.0_f64..1e6, 0.0_f64..1e6, "[a-z%/]{1,4}"), 0..4),
  ) {
    let record = RecordFields {
      texts,
      quantities: quantities
        .into_iter()
        .map(|(name, (a, b, unit))| (name, Quantity::range(a.min(b), a.max(b), unit)))
        .collect(),
    };
    let json = serde_json::to_string(&record).unwrap();
    prop_assert_eq!(serde_json::from_str::<RecordFields>(&json).unwrap(), record);
  }
}

/// D17, end to end on one generated evaluation: judging again under the configuration it was judged with changes
/// nothing. (The scoring properties in `similarity-core` cover judging under another.)
#[test]
fn an_evaluation_judged_again_under_its_own_configuration_is_itself() {
  let judged = |evaluation: &Evaluation| {
    let mut again = evaluation.clone();
    again.rejudge(&schema(), &Policy::new(0.75, None).unwrap());
    again
  };
  let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
  runtime.block_on(async {
    let engine = engine(None, false);
    let record = |name: &str, weight: f64| EvaluateInput {
      record: RecordInput {
        id: Some("a".to_owned()),
        tenant: None,
        fields: vec![("name".to_owned(), name.to_owned())],
        quantities: vec![("weight".to_owned(), Quantity::single(weight, "g"))],
      },
      mode: WriteMode::DryRun,
      threshold: None,
    };
    let mut first = record(NAMES[0], 1.0);
    first.mode = WriteMode::Enforce;
    engine.evaluate(first).await.unwrap();
    for (name, weight) in [(NAMES[1], 1.0), (NAMES[4], 28.0)] {
      let output = engine.evaluate(record(name, weight)).await.unwrap();
      // The response carries the measurements; put them back into the shape the scoring code judges.
      let evaluation = Evaluation {
        comparison: similarity_core::Comparison {
          similarity: output.similarity.unwrap(),
          fields: output.fields.clone(),
        },
        anchor_similarity: None,
        reasons: Vec::new(),
        quantities: output.quantities.clone(),
        measure_changed: output.measure_changed,
      };
      let again = judged(&evaluation);
      assert!((again.comparison.similarity - output.similarity.unwrap()).abs() < 1e-6);
      assert_eq!(
        again.reasons.iter().map(ToString::to_string).collect::<Vec<_>>(),
        output.reasons,
        "{name}"
      );
    }
  });
}
