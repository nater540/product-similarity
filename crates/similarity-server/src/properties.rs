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
//! - an import creates a record or leaves it exactly alone, and what it creates has the original version, where
//!   one was given, as its anchor;
//! - nothing ever happens to a record other than the one a request is about.
//!
//! It runs on the in-memory store and the bag-of-words embedder, so it needs no model and no Qdrant. See
//! `similarity-core`'s `properties` module for the scoring properties and for how failures are saved.

use proptest::prelude::*;
use proptest::test_runner::Config;
use similarity_core::score::Evaluation;
use similarity_core::testing::{WORDS, tiny_embedder};
use similarity_core::vector::cosine;
use similarity_core::{EmbedderOptions, FieldSpec, Policy, Quantity, QuantitySpec, Reason, Schema};
use similarity_store::{MemoryStore, StoredRecord, VectorStore};

use crate::config::ShadowEvents;
use crate::engine::{
  Engine, EngineError, EvaluateInput, FindSimilarInput, ImportInput, Imported, Original, RecordInput, SimilarTo,
  Verdict, WriteMode, store_layout,
};
use crate::offline::calibrate::suggest_threshold;
use crate::offline::pairs::Label;
use crate::offline::{Offline, RecordFields, Suggestion};
use crate::shadow::PolicyShadow;
use crate::worker::testing::{WORD_BAG_DIMENSION, WordBagEmbedder, word_bag};
use crate::worker::{EmbedError, TextEmbedder};

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
  /// A bulk import of one record, with or without the version it was created as.
  Import {
    id: usize,
    tenant: Option<usize>,
    name: usize,
    category: Option<usize>,
    weight: Option<(f64, &'static str)>,
    original: Option<(usize, Option<(f64, &'static str)>)>,
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
  let name = || prop_oneof![9 => 0..NAMES.len() - 1, 1 => Just(NAMES.len() - 1)];
  let import = (
    0..IDS.len(),
    proptest::option::of(0..TENANTS.len()),
    name(),
    proptest::option::of(0..CATEGORIES.len()),
    proptest::option::of(quantity()),
    proptest::option::of((name(), proptest::option::of(quantity()))),
  )
    .prop_map(|(id, tenant, name, category, weight, original)| Request::Import {
      id,
      tenant,
      name,
      category,
      weight,
      original,
    });
  prop_oneof![12 => evaluate, 2 => delete, 3 => import, 1 => search]
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

fn sound_engine<T>(result: Result<T, EngineError>) -> Result<T, TestCaseError> {
  result.map_err(|error| TestCaseError::fail(format!("the engine failed: {error:?}")))
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
/// The `Evaluate` call a request stands for, if it is one.
fn evaluate_input(request: &Request) -> Option<EvaluateInput> {
  let Request::Evaluate {
    id,
    tenant,
    name,
    category,
    description,
    weight,
    thc,
    mode,
    threshold,
  } = request.clone()
  else {
    return None;
  };
  let mut fields = vec![("name".to_owned(), NAMES[name].to_owned())];
  fields.extend(category.map(|index| ("category".to_owned(), CATEGORIES[index].to_owned())));
  fields.extend(description.map(|index| ("description".to_owned(), DESCRIPTIONS[index].to_owned())));
  let mut quantities = Vec::new();
  quantities.extend(weight.map(|(value, unit)| ("weight".to_owned(), Quantity::single(value, unit))));
  quantities.extend(thc.map(|(value, unit)| ("thc".to_owned(), Quantity::single(value, unit))));
  Some(EvaluateInput {
    record: RecordInput {
      id: id.map(|index| IDS[index].to_owned()),
      tenant: tenant.map(|index| TENANTS[index].to_owned()),
      fields,
      quantities,
    },
    mode,
    threshold,
  })
}

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
      weight,
      thc,
      mode,
      threshold,
      ..
    } => {
      let tenant = tenant.map(|index| TENANTS[index].to_owned());
      let input = evaluate_input(request).unwrap();
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

      // A reason about a field or a quantity names an entry of this response that is marked as beyond its limit.
      // The API promises a client that join (`ReasonDetail.subject`).
      for reason in &output.reasons {
        match reason {
          Reason::FieldBelowFloor { field, .. } => {
            prop_assert!(
              output
                .fields
                .iter()
                .any(|score| &score.field == field && score.below_floor)
            );
          }
          Reason::QuantityBeyondFactor { field, .. } | Reason::QuantityIncomparable { field, .. } => {
            prop_assert!(
              output
                .quantities
                .iter()
                .any(|score| &score.field == field && score.beyond_limit)
            );
          }
          Reason::BelowThreshold { .. } | Reason::AnchorDrift { .. } | Reason::MeasureChanged => {
            prop_assert!(reason.subject().is_none());
          }
        }
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
    Request::Import {
      id,
      tenant,
      name,
      category,
      weight,
      original,
    } => {
      let tenant = tenant.map(|index| TENANTS[index].to_owned());
      let mut fields = vec![("name".to_owned(), NAMES[name].to_owned())];
      fields.extend(category.map(|index| ("category".to_owned(), CATEGORIES[index].to_owned())));
      let weigh = |weight: Option<(f64, &str)>| -> Vec<(String, Quantity)> {
        weight
          .map(|(value, unit)| ("weight".to_owned(), Quantity::single(value, unit)))
          .into_iter()
          .collect()
      };
      let input = ImportInput {
        record: RecordInput {
          id: Some(IDS[id].to_owned()),
          tenant: tenant.clone(),
          fields,
          quantities: weigh(weight),
        },
        anchor: original.map(|(name, weight)| Original {
          fields: vec![("name".to_owned(), NAMES[name].to_owned())],
          quantities: weigh(weight),
        }),
      };
      let mut outcomes = sound_engine(engine.import(vec![input]).await)?;
      let outcome = outcomes
        .pop()
        .ok_or_else(|| TestCaseError::fail("one record, no outcome"))?;
      let after = stored(engine).await;
      for (index, (was, is)) in before.iter().zip(&after).enumerate() {
        if index != id {
          prop_assert_eq!(was, is);
        }
      }
      let (was, is) = (before[id].as_ref(), after[id].as_ref());

      let invalid = is_invalid(name, weight, None, None)
        || original.is_some_and(|(name, weight)| is_invalid(name, weight, None, None));
      let owner = was.and_then(|record| record.tenant.as_ref());
      if invalid {
        prop_assert!(matches!(outcome, Err(EngineError::InvalidArgument(_))), "{:?}", outcome);
        prop_assert_eq!(was, is);
      } else if let Some(was) = was {
        // An import never updates. Not the record, not its anchor, not its timestamp.
        prop_assert_eq!(Some(was), is, "an import changed a record that was already stored");
        if owner.is_some_and(|owner| Some(owner) != tenant.as_ref()) {
          prop_assert!(
            matches!(outcome, Err(EngineError::FailedPrecondition(_))),
            "D14: {:?}",
            outcome
          );
        } else {
          prop_assert_eq!(outcome, Ok(Imported::Skipped));
        }
      } else {
        prop_assert_eq!(outcome, Ok(Imported::Created));
        let Some(is) = is else {
          return Err(TestCaseError::fail("created, and nothing is stored"));
        };
        prop_assert_eq!((is.revision, &is.tenant), (1, &tenant));
        let schema = schema();
        let anchor = sound(schema.blocks(WORD_BAG_DIMENSION, &is.anchor))?;
        let current = sound(schema.blocks(WORD_BAG_DIMENSION, &is.vectors.composite))?;
        match original {
          None => {
            prop_assert_eq!(&is.anchor, &is.vectors.composite);
            prop_assert_eq!(&is.anchor_quantities, &is.quantities);
          }
          Some((original_name, original_weight)) => {
            // Schema order: category, description, name. The name is the original's; the category, which the
            // original version here never has, is today's.
            let (Some(Some(kept)), Some(category)) = (anchor.get(2), anchor.first()) else {
              return Err(TestCaseError::fail("the anchor has no name"));
            };
            prop_assert!(
              sound(cosine(kept, &word_bag(NAMES[original_name])))? > 0.9999,
              "the original name"
            );
            prop_assert_eq!(category.is_some(), current.first().is_some_and(Option::is_some));
            let expected = original_weight
              .or(weight)
              .map(|(value, unit)| Quantity::single(value, unit.trim()));
            prop_assert_eq!(
              is.anchor_quantities.get("weight"),
              expected.as_ref(),
              "the first weight it had"
            );
          }
        }
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
      // The kinds and what they are about, not the numbers inside: the similarity was allowed 1e-6 just above.
      let kinds = |reasons: &[Reason]| -> Vec<(&'static str, Option<String>)> {
        reasons
          .iter()
          .map(|reason| (reason.code(), reason.subject().map(str::to_owned)))
          .collect()
      };
      assert_eq!(kinds(&again.reasons), kinds(&output.reasons), "{name}");
    }
  });
}

/// The offline scorer's own model behind the engine's embedder interface, so that one set of weights scores a
/// pair both ways.
struct OfflineModel(std::sync::Arc<Offline>);

impl TextEmbedder for OfflineModel {
  async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
    self
      .0
      .embedder()
      .embed(&texts)
      .map_err(|error| EmbedError::Failed(error.to_string()))
  }

  fn dimension(&self) -> usize {
    self.0.dimension()
  }

  fn fingerprint(&self) -> &str {
    self.0.fingerprint()
  }
}

/// One version of a record: always a name, sometimes a brand, sometimes a weight in one of several units.
fn version() -> impl Strategy<Value = RecordFields> {
  let phrase =
    || proptest::collection::vec(proptest::sample::select(WORDS.to_vec()), 1..=3).prop_map(|words| words.join(" "));
  let weight = (
    proptest::sample::select(vec![0.5, 1.0, 3.5, 28.0]),
    proptest::sample::select(vec!["g", "mg", "oz", "ml"]),
  );
  (phrase(), proptest::option::of(phrase()), proptest::option::of(weight)).prop_map(|(name, brand, weight)| {
    let mut record = RecordFields::default();
    record.texts.insert("name".to_owned(), name);
    if let Some(brand) = brand {
      record.texts.insert("brand".to_owned(), brand);
    }
    if let Some((amount, unit)) = weight {
      record
        .quantities
        .insert("weight".to_owned(), Quantity::single(amount, unit));
    }
    record
  })
}

proptest! {
  #![proptest_config(cases())]

  /// D8, D20: a calibration pair that says what the record was first created as is judged exactly as the server
  /// judges a record imported with that original version and then revised. The pair is how a blank-and-refill
  /// swap gets into a calibration or the verdict snapshot, so the two must not come apart: every measurement,
  /// every reason, and the verdict are the same.
  #[test]
  fn an_anchored_pair_is_judged_as_an_import_followed_by_an_evaluate(
    anchor in version(),
    before in version(),
    after in version(),
  ) {
    let field = |name: &str, weight: f32, required: bool, reject_below: Option<f32>| FieldSpec {
      name: name.to_owned(),
      weight,
      required,
      reject_below,
    };
    let schema = Schema::new(vec![field("name", 2.0, true, Some(0.5)), field("brand", 1.0, false, None)])
      .unwrap()
      .with_quantities(vec![QuantitySpec {
        name: "weight".to_owned(),
        reject_above_factor: Some(4.0),
        reject_incomparable: true,
      }])
      .unwrap();
    let mut policy = Policy::new(0.8, Some(0.7)).unwrap();
    policy.reject_on_measure_change = true;
    let options = EmbedderOptions { max_tokens: 16, batch_size: 8, output_dim: None };
    let offline = std::sync::Arc::new(Offline::new(tiny_embedder(&options), schema.clone(), policy));

    let expected = offline
      .compare(&before, &after, Some(&anchor))
      .map_err(|error| TestCaseError::fail(format!("the offline scorer failed: {error:#}")))?;

    let embedder = OfflineModel(std::sync::Arc::clone(&offline));
    let (layout, dimension) = store_layout(&schema, &embedder);
    let engine = Engine::new(MemoryStore::new(layout, dimension), embedder, schema, policy);
    let fields = |record: &RecordFields| -> Vec<(String, String)> {
      record.texts.iter().map(|(name, text)| (name.clone(), text.clone())).collect()
    };
    let quantities = |record: &RecordFields| -> Vec<(String, Quantity)> {
      record.quantities.iter().map(|(name, quantity)| (name.clone(), quantity.clone())).collect()
    };
    let input = |record: &RecordFields| RecordInput {
      id: Some("p".to_owned()),
      tenant: None,
      fields: fields(record),
      quantities: quantities(record),
    };
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let output = runtime.block_on(async {
      let original = Original { fields: fields(&anchor), quantities: quantities(&anchor) };
      let imported = engine
        .import(vec![ImportInput { record: input(&before), anchor: Some(original) }])
        .await?;
      assert!(matches!(imported[..], [Ok(Imported::Created)]), "{imported:?}");
      engine
        .evaluate(EvaluateInput { record: input(&after), mode: WriteMode::DryRun, threshold: None })
        .await
    });
    let output = sound_engine(output)?;

    let close = |found: Option<f32>, wanted: Option<f32>| match (found, wanted) {
      (Some(found), Some(wanted)) => (found - wanted).abs() < 1e-5,
      (found, wanted) => found == wanted,
    };
    prop_assert!(close(output.similarity, Some(expected.comparison.similarity)));
    prop_assert!(close(output.anchor_similarity, expected.anchor_similarity));
    prop_assert_eq!(output.fields.len(), expected.comparison.fields.len());
    for (found, wanted) in output.fields.iter().zip(&expected.comparison.fields) {
      prop_assert_eq!((&found.field, found.change, found.below_floor), (&wanted.field, wanted.change, wanted.below_floor));
      prop_assert!(close(found.similarity, wanted.similarity), "{}: {:?} and {:?}", found.field, found.similarity, wanted.similarity);
    }
    prop_assert_eq!(&output.quantities, &expected.quantities);
    prop_assert_eq!(output.measure_changed, expected.measure_changed);
    let kinds = |reasons: &[Reason]| -> Vec<(&'static str, Option<String>)> {
      reasons.iter().map(|reason| (reason.code(), reason.subject().map(str::to_owned))).collect()
    };
    prop_assert_eq!(kinds(&output.reasons), kinds(&expected.reasons));
    prop_assert_eq!(output.verdict == Verdict::Rejected, !expected.accepted());
  }
}

/// The candidates a shadow is tried with: one that disagrees with the live policy as often as it can in each
/// direction, one with other weights and limits, and one that cannot judge anything at all.
fn candidate(kind: usize) -> PolicyShadow {
  let field = |name: &str, weight: f32, reject_below: Option<f32>| FieldSpec {
    name: name.to_owned(),
    weight,
    required: name == "name",
    reject_below,
  };
  let quantity = |name: &str, reject_above_factor: Option<f64>| QuantitySpec {
    name: name.to_owned(),
    reject_above_factor,
    reject_incomparable: true,
  };
  let reweighted = Schema::new(vec![
    field("name", 1.0, None),
    field("category", 1.0, Some(0.9)),
    field("description", 5.0, None),
  ])
  .unwrap()
  .with_quantities(vec![quantity("weight", None), quantity("thc", Some(1.5))])
  .unwrap();
  let mut strict = Policy::new(0.999, Some(0.999)).unwrap();
  strict.reject_on_measure_change = true;
  let (schema, policy, wrong_by) = match kind {
    0 => (schema(), strict, 0),
    1 => (schema(), Policy::new(-1.0, None).unwrap(), 0),
    2 => (reweighted, Policy::new(0.75, Some(0.5)).unwrap(), 0),
    _ => (schema(), strict, 1),
  };
  PolicyShadow::new(
    "candidate",
    schema,
    policy,
    WORD_BAG_DIMENSION + wrong_by,
    ShadowEvents::All,
  )
}

proptest! {
  #![proptest_config(cases())]

  /// A shadow decides nothing: with any candidate watching, every request in any sequence gets the answer it
  /// gets with nobody watching, and leaves the store as it leaves it then. The write rules hold beside it too.
  #[test]
  fn a_shadow_never_changes_an_answer_or_a_write(
    requests in proptest::collection::vec(request(), 1..40),
    anchor_threshold in proptest::option::of(Just(0.5_f32)),
    kind in 0_usize..4,
  ) {
    let unwatched = engine(anchor_threshold, false);
    let shadow = std::sync::Arc::new(candidate(kind));
    let watched = engine(anchor_threshold, false).with_observer(shadow.clone());
    let timeless = |records: Vec<Option<StoredRecord>>| -> Vec<Option<StoredRecord>> {
      records
        .into_iter()
        .map(|record| record.map(|record| StoredRecord { updated_at: 0, ..record }))
        .collect()
    };
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    runtime.block_on(async {
      let mut judged = 0_u64;
      for (index, request) in requests.iter().enumerate() {
        if let Some(input) = evaluate_input(request) {
          let alone = unwatched.evaluate(input.clone()).await;
          let beside = watched.evaluate(input).await;
          prop_assert_eq!(&alone, &beside, "request {}, {:?}", index, request);
          let revised = alone.is_ok_and(|output| matches!(output.verdict, Verdict::Accepted | Verdict::Rejected));
          judged += u64::from(revised);
        } else {
          for engine in [&unwatched, &watched] {
            step(engine, request)
              .await
              .map_err(|error| TestCaseError::fail(format!("request {index}, {request:?}: {error}")))?;
          }
        }
        prop_assert_eq!(
          timeless(stored(&unwatched).await),
          timeless(stored(&watched).await),
          "request {}, {:?}",
          index,
          request
        );
      }
      // And it was watching: every revision was put to the candidate, which judged it or failed to.
      let summary = shadow.summary();
      prop_assert_eq!(summary.judged + summary.failed, judged);
      prop_assert_eq!(summary.failed > 0, kind == 3 && judged > 0);
      prop_assert_eq!(
        summary.agreed + summary.only_shadow_rejected + summary.only_live_rejected,
        summary.judged
      );
      Ok::<(), TestCaseError>(())
    })?;
  }
}

proptest! {
  #![proptest_config(cases())]

  /// Whatever was promoted and retired in whatever order, the model in service is a promotion whose model has
  /// not been retired since, and the latest such; and a model's retirement stands exactly until it is promoted
  /// again.
  #[test]
  fn the_model_in_service_is_the_latest_promotion_not_retired_since(
    history in proptest::collection::vec((any::<bool>(), 0_usize..3), 0..24),
  ) {
    use crate::registry::testing::{Held, PAIRS, calibration, promoted, written};
    use crate::registry::{Entry, HOLDOUT, Registry, Retired};

    const DIGESTS: [&str; 3] = ["a", "b", "c"];
    let entries: Vec<Entry> = history
      .iter()
      .enumerate()
      .map(|(index, (promote, model))| {
        let digest = DIGESTS[*model];
        if *promote {
          let text = calibration(&format!("{digest}-d8-t16"), Some(HOLDOUT), "aa", &PAIRS);
          let mut entry = promoted(digest, 0.75, &Held::of(&text, "holdout.ndjson"));
          entry.written = written(index as u64);
          Entry::Promoted(Box::new(entry))
        } else {
          Entry::Retired(Retired {
            written: written(index as u64),
            digest: digest.to_owned(),
            reason: "retired".to_owned(),
          })
        }
      })
      .collect();
    let registry = Registry { entries };

    // Said the long way round: for each model, what happened to it last.
    let last = |digest: &str, promote: bool| {
      history
        .iter()
        .rposition(|(promoted, model)| *promoted == promote && DIGESTS[*model] == digest)
    };
    let standing: Vec<usize> = DIGESTS
      .iter()
      .filter_map(|digest| match (last(digest, true), last(digest, false)) {
        (Some(promoted), Some(retired)) if retired > promoted => None,
        (promoted, _) => promoted,
      })
      .collect();
    let expected = standing.iter().max().map(|index| (DIGESTS[history[*index].1], *index as u64));
    let current = registry.current().map(|promoted| (promoted.digest.as_str(), promoted.written.at));
    prop_assert_eq!(current, expected);

    for digest in DIGESTS {
      let retired = matches!((last(digest, true), last(digest, false)), (promoted, Some(retired)) if promoted.is_none_or(|promoted| retired > promoted));
      prop_assert_eq!(registry.retirement(digest).is_some(), retired, "{}", digest);
      prop_assert_eq!(registry.knows(digest), last(digest, true).is_some() || last(digest, false).is_some());
    }
  }
}
