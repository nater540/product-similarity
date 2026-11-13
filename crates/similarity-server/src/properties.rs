//! Properties of the service, checked on generated inputs.
//!
//! The engine's tests are scenarios someone thought of. This drives the engine with random sequences of requests
//! (every write mode, with and without quantities and brands, valid and not) and after every single request
//! checks the rules the product stands on, whatever came before:
//!
//! - a rejected revision, a dry run, and a failed request leave the stored record exactly as it was (D7);
//! - an accepted write never replaces a value the anchor holds, text or quantity, and gives it only the first
//!   value of something it never had (D8, D16);
//! - a forced write makes the record its own anchor;
//! - the revision counter moves by exactly one, exactly when something was stored;
//! - a record with a brand is touched only by a request that names it (D14);
//! - an import creates a record or leaves it exactly alone, and what it creates has the original version, where
//!   one was given, as its anchor;
//! - nothing ever happens to a record other than the one a request is about;
//! - two replicas asked about a record at the same moment end as they would have had the two requests arrived
//!   one after the other (D10);
//! - within one process, a record is held by one request at a time, the one that came first, a request never
//!   waits for a request for another record, and a record has a lock only while somebody is at it (D41).
//!
//! - a restore of anchors from a backup changes a record only if it has no history of its own and is what the
//!   backup was taken of, and a store that is lost, imported again, and restored has every anchor it had (D43);
//!
//! And the audit, which vouches for what the store holds, is held to what an import and the engine really do:
//!
//! - an export that was just imported audits clean, whatever is wrong with its lines;
//! - one change to the store or to the export is found, as the kind of change it is, and nothing else is;
//! - everything an audit looks at is counted once, whatever the size of a page.
//!
//! It runs on the in-memory store and the bag-of-words embedder, so it needs no model and no Qdrant. See
//! `similarity-core`'s `properties` module for the scoring properties and for how failures are saved.

use proptest::prelude::*;
use proptest::test_runner::Config;
use similarity_core::matching::Decision;
use similarity_core::score::Evaluation;
use similarity_core::testing::{WORDS, tiny_embedder};
use similarity_core::vector::cosine;
use similarity_core::{EmbedderOptions, FieldSpec, Policy, Quantity, QuantitySpec, Reason, Schema};
use similarity_store::{Entry, MemoryStore, Read, Scan, StoredRecord, VectorStore};

use crate::engine::racing::Raced;
use crate::locks::Locks;
use crate::locks::testing::Asking;

use crate::audit::{self, Difference, Finding, Kind};
use crate::config::ShadowEvents;
use crate::engine::{
  Engine, EngineError, EvaluateInput, EvaluateOutput, FindSimilarInput, ImportInput, Imported, KeptAnchor, MatchInput,
  Original, RecordInput, Restored, SimilarTo, Verdict, WriteMode, store_layout,
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
const BRAND_IDS: [&str; 2] = ["brand-1", "brand-2"];
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
    brand_id: Option<usize>,
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
    brand_id: Option<usize>,
  },
  /// A bulk import of one record, with or without the version it was created as.
  Import {
    id: usize,
    brand_id: Option<usize>,
    name: usize,
    category: Option<usize>,
    weight: Option<(f64, &'static str)>,
    original: Option<(usize, Option<(f64, &'static str)>)>,
  },
  Search {
    id: usize,
  },
  /// A menu item from elsewhere, to be matched against whatever is stored.
  Match {
    name: usize,
    category: Option<usize>,
    brand_id: Option<usize>,
    threshold: Option<f32>,
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
      proptest::option::of(0..BRAND_IDS.len()),
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
      |((id, brand_id, name, category, description), (weight, thc, mode, threshold))| Request::Evaluate {
        id,
        brand_id,
        name,
        category,
        description,
        weight,
        thc,
        mode,
        threshold,
      },
    );
  let delete = (0..IDS.len(), proptest::option::of(0..BRAND_IDS.len()))
    .prop_map(|(id, brand_id)| Request::Delete { id, brand_id });
  let search = (0..IDS.len()).prop_map(|id| Request::Search { id });
  let name = || prop_oneof![9 => 0..NAMES.len() - 1, 1 => Just(NAMES.len() - 1)];
  let import = (
    0..IDS.len(),
    proptest::option::of(0..BRAND_IDS.len()),
    name(),
    proptest::option::of(0..CATEGORIES.len()),
    proptest::option::of(quantity()),
    proptest::option::of((name(), proptest::option::of(quantity()))),
  )
    .prop_map(|(id, brand_id, name, category, weight, original)| Request::Import {
      id,
      brand_id,
      name,
      category,
      weight,
      original,
    });
  let matched = (
    prop_oneof![9 => 0..NAMES.len() - 1, 1 => Just(NAMES.len() - 1)],
    proptest::option::of(0..CATEGORIES.len()),
    proptest::option::of(0..BRAND_IDS.len()),
    prop_oneof![2 => Just(None), 1 => Just(Some(0.5_f32)), 1 => Just(Some(7.0_f32))],
  )
    .prop_map(|(name, category, brand_id, threshold)| Request::Match {
      name,
      category,
      brand_id,
      threshold,
    });
  prop_oneof![12 => evaluate, 2 => delete, 3 => import, 1 => search, 2 => matched]
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
    brand_id,
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
      brand_id: brand_id.map(|index| BRAND_IDS[index].to_owned()),
      fields,
      quantities,
    },
    mode,
    threshold,
  })
}

/// The `Import` call a request stands for, if it is one.
fn import_input(request: &Request) -> Option<ImportInput> {
  let Request::Import {
    id,
    brand_id,
    name,
    category,
    weight,
    original,
  } = request.clone()
  else {
    return None;
  };
  let mut fields = vec![("name".to_owned(), NAMES[name].to_owned())];
  fields.extend(category.map(|index| ("category".to_owned(), CATEGORIES[index].to_owned())));
  let weigh = |weight: Option<(f64, &str)>| -> Vec<(String, Quantity)> {
    weight
      .map(|(value, unit)| ("weight".to_owned(), Quantity::single(value, unit)))
      .into_iter()
      .collect()
  };
  Some(ImportInput {
    record: RecordInput {
      id: Some(IDS[id].to_owned()),
      brand_id: brand_id.map(|index| BRAND_IDS[index].to_owned()),
      fields,
      quantities: weigh(weight),
    },
    anchor: original.map(|(name, weight)| Original {
      fields: vec![("name".to_owned(), NAMES[name].to_owned())],
      quantities: weigh(weight),
    }),
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
      brand_id,
      name,
      weight,
      thc,
      mode,
      threshold,
      ..
    } => {
      let brand_id = brand_id.map(|index| BRAND_IDS[index].to_owned());
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

      let brand = was.and_then(|record| record.brand_id.as_ref());
      let wrong_brand = brand.is_some_and(|brand| Some(brand) != brand_id.as_ref());
      let output = match outcome {
        Err(error) => {
          prop_assert_eq!(was, is, "a failed request wrote something: {:?}", error);
          if is_invalid(name, weight, thc, threshold) {
            prop_assert!(matches!(error, EngineError::InvalidArgument(_)), "{:?}", error);
          } else {
            prop_assert!(wrong_brand, "an acceptable request was refused: {:?}", error);
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
        !wrong_brand,
        "D14: a record with a brand was evaluated by a request that does not name it"
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
      prop_assert_eq!(&is.brand_id, &brand_id);
      match (was, mode) {
        // A new record and a forced one are their own point of reference.
        (None, _) | (Some(_), WriteMode::Force) => {
          prop_assert_eq!(&is.anchor, &is.vectors.composite);
          prop_assert_eq!(&is.anchor_quantities, &is.quantities);
        }
        (Some(was), _) => check_anchor_kept(was, is)?,
      }
    }
    Request::Delete { id, brand_id } => {
      let brand_id = brand_id.map(|index| BRAND_IDS[index].to_owned());
      let outcome = engine.delete(IDS[id].to_owned(), brand_id.clone()).await;
      let after = stored(engine).await;
      for (index, (was, is)) in before.iter().zip(&after).enumerate() {
        if index != id {
          prop_assert_eq!(was, is);
        }
      }
      let brand = before[id].as_ref().and_then(|record| record.brand_id.as_ref());
      if brand.is_some_and(|brand| Some(brand) != brand_id.as_ref()) {
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
      brand_id,
      name,
      weight,
      original,
      ..
    } => {
      let brand_id = brand_id.map(|index| BRAND_IDS[index].to_owned());
      let input = import_input(request).unwrap();
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
      let brand = was.and_then(|record| record.brand_id.as_ref());
      if invalid {
        prop_assert!(matches!(outcome, Err(EngineError::InvalidArgument(_))), "{:?}", outcome);
        prop_assert_eq!(was, is);
      } else if let Some(was) = was {
        // An import never updates. Not the record, not its anchor, not its timestamp.
        prop_assert_eq!(Some(was), is, "an import changed a record that was already stored");
        if brand.is_some_and(|brand| Some(brand) != brand_id.as_ref()) {
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
        prop_assert_eq!((is.revision, &is.brand_id), (1, &brand_id));
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
    Request::Match {
      name,
      category,
      brand_id,
      threshold,
    } => {
      let mut fields = vec![("name".to_owned(), NAMES[name].to_owned())];
      fields.extend(category.map(|index| ("category".to_owned(), CATEGORIES[index].to_owned())));
      let brand_id = brand_id.map(|index| BRAND_IDS[index].to_owned());
      let outcome = engine
        .match_menu_item(MatchInput {
          menu_item: RecordInput {
            id: None,
            brand_id: None,
            fields,
            quantities: Vec::new(),
          },
          limit: 0,
          brand_id: brand_id.clone(),
          threshold,
        })
        .await;
      // A match is a read, whatever it finds and however it ends.
      prop_assert_eq!(&before, &stored(engine).await, "a match wrote something");
      let invalid = NAMES[name].trim().is_empty() || threshold.is_some_and(|threshold| threshold > 1.0);
      match outcome {
        Err(error) => prop_assert!(
          invalid && matches!(error, EngineError::InvalidArgument(_)),
          "{:?}",
          error
        ),
        Ok(output) => {
          prop_assert!(!invalid, "an unusable menu item was matched");
          // Every stored record the request may see is a candidate (there are fewer than the limit), best first.
          let visible = before
            .iter()
            .flatten()
            .filter(|record| brand_id.is_none() || record.brand_id == brand_id)
            .count();
          prop_assert_eq!(output.candidates.len(), visible);
          prop_assert_eq!(output.fetched, visible);
          for pair in output.candidates.windows(2) {
            prop_assert!(pair[0].comparison.similarity >= pair[1].comparison.similarity);
          }
          for candidate in &output.candidates {
            prop_assert!(candidate.comparison.fields.iter().all(|field| !field.below_floor));
            prop_assert!(candidate.quantities.iter().all(|quantity| !quantity.beyond_limit));
          }
          let best = output
            .candidates
            .first()
            .map(|candidate| candidate.comparison.similarity);
          let expected = match (threshold, best) {
            (None, _) => Decision::NotJudged,
            (Some(threshold), Some(best)) if best >= threshold => Decision::Matched,
            (Some(_), _) => Decision::NoMatch,
          };
          prop_assert_eq!(output.decision, expected);
        }
      }
    }
    Request::Search { id } => {
      let outcome = engine
        .find_similar(FindSimilarInput {
          query: SimilarTo::Id(IDS[id].to_owned()),
          limit: 0,
          brand_id: None,
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
        brand_id: None,
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
      brand_id: None,
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

/// What a request that can write was answered. A search and a match write nothing and are not asked here.
#[derive(Debug, Clone, PartialEq)]
enum Answer {
  Evaluated(Box<Result<EvaluateOutput, EngineError>>),
  Imported(Result<Vec<Result<Imported, EngineError>>, EngineError>),
  Deleted(Result<bool, EngineError>),
  NotAsked,
}

async fn ask<S: VectorStore>(engine: &Engine<S, WordBagEmbedder>, request: &Request) -> Answer {
  if let Some(input) = evaluate_input(request) {
    return Answer::Evaluated(Box::new(engine.evaluate(input).await));
  }
  if let Some(input) = import_input(request) {
    return Answer::Imported(engine.import(vec![input]).await);
  }
  match request {
    Request::Delete { id, brand_id } => {
      let brand_id = brand_id.map(|index| BRAND_IDS[index].to_owned());
      Answer::Deleted(engine.delete(IDS[*id].to_owned(), brand_id).await)
    }
    _ => Answer::NotAsked,
  }
}

fn engine_on<S: VectorStore>(store: S, anchor_threshold: Option<f32>) -> Engine<S, WordBagEmbedder> {
  Engine::new(
    store,
    WordBagEmbedder::new("word-bag-v1"),
    schema(),
    Policy::new(0.75, anchor_threshold).unwrap(),
  )
}

fn empty_store() -> MemoryStore {
  let (layout, dimension) = store_layout(&schema(), &WordBagEmbedder::new("word-bag-v1"));
  MemoryStore::new(layout, dimension)
}

/// Every record the requests can be about, without the time it was written: two runs of one history differ in
/// nothing else.
async fn timeless(store: &MemoryStore) -> Vec<Option<StoredRecord>> {
  let mut records = Vec::new();
  for id in IDS {
    let record = store.get(id).await.unwrap();
    records.push(record.map(|record| StoredRecord {
      updated_at: 0,
      ..record
    }));
  }
  records
}

/// `history`, then `first`, then `second`, each finished before the next begins: both answers, and what is stored.
async fn one_after_the_other(
  history: &[Request],
  first: &Request,
  second: &Request,
  anchor_threshold: Option<f32>,
) -> (Answer, Answer, Vec<Option<StoredRecord>>) {
  let engine = engine_on(empty_store(), anchor_threshold);
  // Boxed, here and below: a request's future is large, and a test thread's stack is not.
  for request in history {
    Box::pin(ask(&engine, request)).await;
  }
  let first = Box::pin(ask(&engine, first)).await;
  let second = Box::pin(ask(&engine, second)).await;
  (first, second, timeless(engine.store()).await)
}

proptest! {
  #![proptest_config(cases())]

  /// Two replicas share a store and no lock (D10). Whatever each is asked at the same moment, with one of them
  /// reading a record before the other writes and going to write after it, what comes of it is what comes of
  /// the two requests arriving one after the other, in one order or the other: both answers, and what is stored.
  /// No request is judged against a record that was replaced under it and then written anyway.
  ///
  /// One answer is allowed to differ, and only that one: two deletes of one record at once are both told that
  /// there was a record, where one after the other the second is told there was none. The store says whether a
  /// record is gone, not who removed it.
  #[test]
  fn two_replicas_asked_at_once_end_as_if_asked_one_after_the_other(
    // What came before, and then the two requests that arrive together: the last two of the list. (One list,
    // because a generated request is a large thing to hold three of at once on a test thread's stack.)
    requests in proptest::collection::vec(request(), 2..14),
    // An import reads once before it embeds and once when it goes to write: the other replica gets in after
    // either.
    late in any::<bool>(),
    anchor_threshold in proptest::option::of(Just(0.5_f32)),
  ) {
    let Some((there_asked, history)) = requests.split_last() else {
      return Err(TestCaseError::reject("two requests are needed to arrive together"));
    };
    let Some((here_asked, history)) = history.split_last() else {
      return Err(TestCaseError::reject("two requests are needed to arrive together"));
    };
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    runtime.block_on(async {
      let store = std::sync::Arc::new(empty_store());
      let there = std::sync::Arc::new(engine_on(std::sync::Arc::clone(&store), anchor_threshold));
      let here = engine_on(Raced::over(std::sync::Arc::clone(&store)), anchor_threshold);
      for request in history {
        Box::pin(ask(&*there, request)).await;
      }

      let answered = std::sync::Arc::new(std::sync::Mutex::new(None));
      let skipped = usize::from(late && matches!(here_asked, Request::Import { .. }));
      here.store().after_read(skipped, {
        let (there, asked, answered) = (std::sync::Arc::clone(&there), there_asked.clone(), std::sync::Arc::clone(&answered));
        async move {
          let answer = Box::pin(ask(&*there, &asked)).await;
          *answered.lock().unwrap() = Some(answer);
        }
      });
      let here_answered = Box::pin(ask(&here, here_asked)).await;
      // A request that was refused before it read anything let nobody in: the other replica is simply next.
      if let Some(still_waiting) = here.store().waiting() {
        still_waiting.await;
      }
      let there_answered = answered.lock().unwrap().take().unwrap();
      let at_once = (here_answered, there_answered, timeless(&store).await);

      let here_first = Box::pin(one_after_the_other(history, here_asked, there_asked, anchor_threshold)).await;
      let there_first = Box::pin(one_after_the_other(history, there_asked, here_asked, anchor_threshold)).await;
      let there_first = (there_first.1, there_first.0, there_first.2);
      let both_deleted = (Answer::Deleted(Ok(true)), Answer::Deleted(Ok(true)));
      let two_deletes_of_one_record = (&at_once.0, &at_once.1) == (&both_deleted.0, &both_deleted.1)
        && at_once.2 == here_first.2
        && here_first.0 == both_deleted.0
        && there_first.1 == both_deleted.1;
      prop_assert!(
        at_once == here_first || at_once == there_first || two_deletes_of_one_record,
        "at once: {:#?}\nhere first: {:#?}\nthere first: {:#?}",
        at_once,
        here_first,
        there_first
      );
      Ok::<(), TestCaseError>(())
    })?;
  }
}

/// What happens at the record locks: a request for a record arrives, or one of those there leaves, whether it
/// holds its record (it is done) or waits for it (its caller gave up).
#[derive(Debug, Clone, Copy)]
enum AtTheLocks {
  Arrives(usize),
  Leaves(usize),
}

fn at_the_locks() -> impl Strategy<Value = AtTheLocks> {
  prop_oneof![
    3 => (0..LOCKED.len()).prop_map(AtTheLocks::Arrives),
    2 => any::<usize>().prop_map(AtTheLocks::Leaves),
  ]
}

/// The records requests arrive for. Few, so that they meet.
const LOCKED: [&str; 3] = ["a", "b", "c"];

proptest! {
  #![proptest_config(cases())]

  /// Whoever arrives and leaves, in whatever order (D41): each record that anybody is at is held by exactly one
  /// request, the one that came first of those still there, so nobody waits for a record that is free or for a
  /// request for another record; and the table holds those records and no others.
  #[test]
  fn a_record_is_held_by_the_first_request_still_there_and_locked_only_while_somebody_is(
    steps in prop::collection::vec(at_the_locks(), 0..80),
  ) {
    let locks = Locks::default();
    // Everyone at a lock, in the order they came.
    let mut there: Vec<(&str, Asking<'_>)> = Vec::new();
    for step in steps {
      match step {
        AtTheLocks::Arrives(record) => {
          let id = LOCKED.get(record).copied().unwrap_or("a");
          there.push((id, Asking::new(&locks, id)));
        }
        AtTheLocks::Leaves(which) if !there.is_empty() => {
          drop(there.remove(which % there.len()));
        }
        AtTheLocks::Leaves(_) => {}
      }
      let mut seen: Vec<&str> = Vec::new();
      for (id, asking) in &mut there {
        let first = !seen.contains(id);
        prop_assert_eq!(asking.holds(), first, "`{}`, first there: {}", id, first);
        if first {
          seen.push(id);
        }
      }
      prop_assert_eq!(locks.in_use(), seen.len());
    }
    drop(there);
    prop_assert_eq!(locks.in_use(), 0);
  }
}

/// The ids an audited catalog is made of. A few, so that exports repeat an id and a store and an export overlap.
const AUDIT_IDS: [&str; 6] = ["a", "b", "c", "d", "e", "f"];

/// One line of a generated export, as indexes into the lists above.
#[derive(Debug, Clone)]
struct ExportLine {
  id: usize,
  brand_id: Option<usize>,
  name: usize,
  category: Option<usize>,
  weight: Option<(f64, &'static str)>,
  flaw: Flaw,
}

/// What is wrong with a line beyond what its values make wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flaw {
  None,
  /// A key an export line may not have. The line still says which record it is about.
  StrayKey,
  /// Not JSON, and so about no record anyone can name.
  NotJson,
}

impl ExportLine {
  fn json(&self) -> String {
    if self.flaw == Flaw::NotJson {
      return "{ this is not a record".to_owned();
    }
    let mut record = serde_json::json!({"name": NAMES[self.name]});
    if let Some(category) = self.category {
      record["category"] = CATEGORIES[category].into();
    }
    if let Some((value, unit)) = self.weight {
      record["weight"] = serde_json::json!({"value": value, "unit": unit});
    }
    let mut line = serde_json::json!({"id": AUDIT_IDS[self.id], "record": record});
    if let Some(brand_id) = self.brand_id {
      line["brand_id"] = BRAND_IDS[brand_id].into();
    }
    if self.flaw == Flaw::StrayKey {
      line["tenant"] = "brand-1".into();
    }
    line.to_string()
  }
}

fn export_text(lines: &[ExportLine]) -> String {
  lines.iter().map(|line| line.json() + "\n").collect()
}

/// Any line: mostly a record, sometimes one an import refuses (a blank name, an amount below zero, a blank unit,
/// a stray key, no JSON at all).
fn export_line() -> impl Strategy<Value = ExportLine> {
  (
    0..AUDIT_IDS.len(),
    proptest::option::of(0..BRAND_IDS.len()),
    prop_oneof![9 => 0..NAMES.len() - 1, 1 => Just(NAMES.len() - 1)],
    proptest::option::of(0..CATEGORIES.len()),
    proptest::option::of(quantity()),
    prop_oneof![10 => Just(Flaw::None), 1 => Just(Flaw::StrayKey), 1 => Just(Flaw::NotJson)],
  )
    .prop_map(|(id, brand_id, name, category, weight, flaw)| ExportLine {
      id,
      brand_id,
      name,
      category,
      weight,
      flaw,
    })
}

/// A catalog of distinct records, every one of which an import takes.
fn catalog() -> impl Strategy<Value = Vec<ExportLine>> {
  let record = (
    proptest::option::of(0..BRAND_IDS.len()),
    0..NAMES.len() - 1,
    proptest::option::of(0..CATEGORIES.len()),
    proptest::option::of((
      prop_oneof![Just(1.0), Just(3.5), Just(28.0)],
      prop_oneof![Just("g"), Just("mg")],
    )),
  );
  proptest::collection::vec(record, 2..AUDIT_IDS.len()).prop_map(|records| {
    records
      .into_iter()
      .enumerate()
      .map(|(id, (brand_id, name, category, weight))| ExportLine {
        id,
        brand_id,
        name,
        category,
        weight,
        flaw: Flaw::None,
      })
      .collect()
  })
}

async fn import_export<S: VectorStore>(engine: &Engine<S, WordBagEmbedder>, export: &str) -> crate::import::Summary {
  let options = crate::import::Options {
    batch_records: 3,
    stop_on_error: false,
  };
  crate::import::run(engine, export.as_bytes(), options, |_| {}, |_| {})
    .await
    .unwrap()
}

async fn audited<S: Scan>(store: &S, export: &str, options: audit::Options) -> (audit::Summary, Vec<Finding>) {
  let mut findings = Vec::new();
  let summary = audit::run(
    store,
    &schema(),
    export.as_bytes(),
    options,
    |finding| findings.push(finding.clone()),
    |_| {},
  )
  .await
  .unwrap();
  (summary, findings)
}

const WHOLE_STORE: audit::Options = audit::Options {
  page_records: 1000,
  exported_at: None,
};

/// One thing done to a catalog that was imported, or to the export it was imported from.
#[derive(Debug, Clone, Copy)]
enum Change {
  /// The record is deleted from the store.
  Deleted,
  /// The stored record is given another name by a forced write.
  Renamed { to: usize },
  /// The stored record is given another amount by a forced write: twice what it weighed, or a gram if it weighed
  /// nothing.
  Reweighed,
  /// The stored record is put under another brand, or under none, behind the engine's back.
  Rebranded { to: Option<usize> },
  /// A record the export does not have is stored.
  Extra,
  /// The record's line is taken out of the export.
  Unlisted,
  /// The record's line is written another way: other whitespace, and its amount in another unit.
  Rewritten,
}

fn change() -> impl Strategy<Value = Change> {
  prop_oneof![
    Just(Change::Deleted),
    (0..NAMES.len() - 1).prop_map(|to| Change::Renamed { to }),
    Just(Change::Reweighed),
    proptest::option::of(0..BRAND_IDS.len()).prop_map(|to| Change::Rebranded { to }),
    Just(Change::Extra),
    Just(Change::Unlisted),
    Just(Change::Rewritten),
  ]
}

/// What a finding says, without where and when: its kind, the record, and what about the record.
type Told = (Kind, Option<String>, Option<String>, Option<Difference>);

fn told(findings: &[Finding]) -> Vec<Told> {
  findings
    .iter()
    .map(|finding| {
      (
        finding.kind,
        finding.id.clone(),
        finding.field.clone().or_else(|| finding.quantity.clone()),
        finding.difference,
      )
    })
    .collect()
}

proptest! {
  #![proptest_config(cases())]

  /// What an import stores is what an audit expects to find. The two read an export by one rule (`prepared`) and
  /// the engine by its own, and this is what holds the three together: whatever the lines of an export are, valid
  /// or not, repeated or not, importing it into an empty store leaves nothing for an audit to find but the lines
  /// themselves.
  #[test]
  fn an_export_audits_clean_against_what_importing_it_stored(
    lines in proptest::collection::vec(export_line(), 0..14),
    page_records in 1..5_usize,
  ) {
    let export = export_text(&lines);
    let engine = engine(None, false);
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let (imported, (summary, findings)) = runtime.block_on(async {
      let imported = import_export(&engine, &export).await;
      let options = audit::Options { page_records, exported_at: None };
      (imported, audited(engine.store(), &export, options).await)
    });

    for finding in &findings {
      prop_assert!(matches!(finding.kind, Kind::InvalidLine | Kind::DuplicateId), "{finding:?}");
    }
    prop_assert_eq!(summary.export_records, imported.imported);
    prop_assert_eq!(summary.invalid_lines + summary.duplicate_ids, imported.failed + imported.skipped);
    prop_assert_eq!(summary.agree, imported.imported);
    prop_assert_eq!(
      (summary.differ, summary.not_stored, summary.not_in_export, summary.stored_records),
      (0, 0, 0, imported.imported)
    );
  }

  /// One change, one finding of the kind the change is, about the record that was changed: and a line written
  /// another way is no change at all.
  #[test]
  fn one_change_to_the_store_or_the_export_is_found_as_that_kind_and_nothing_else_is(
    records in catalog(),
    at in any::<proptest::sample::Index>(),
    change in change(),
  ) {
    let chosen = at.index(records.len());
    let record = records[chosen].clone();
    let id = AUDIT_IDS[record.id].to_owned();
    let engine = engine(None, false);
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();

    let mut export_lines = records.clone();
    let expected: Vec<Told> = runtime.block_on(async {
      import_export(&engine, &export_text(&records)).await;
      let rewrite = |name: usize, weight: Option<(f64, &'static str)>| {
        let mut fields = vec![("name".to_owned(), NAMES[name].to_owned())];
        fields.extend(record.category.map(|index| ("category".to_owned(), CATEGORIES[index].to_owned())));
        EvaluateInput {
          record: RecordInput {
            id: Some(id.clone()),
            brand_id: record.brand_id.map(|index| BRAND_IDS[index].to_owned()),
            fields,
            quantities: weight
              .map(|(value, unit)| ("weight".to_owned(), Quantity::single(value, unit)))
              .into_iter()
              .collect(),
          },
          mode: WriteMode::Force,
          threshold: None,
        }
      };
      match change {
        Change::Deleted => {
          let brand_id = record.brand_id.map(|index| BRAND_IDS[index].to_owned());
          engine.delete(id.clone(), brand_id).await.unwrap();
          vec![(Kind::NotStored, Some(id.clone()), None, None)]
        }
        Change::Renamed { to } => {
          engine.evaluate(rewrite(to, record.weight)).await.unwrap();
          if to == record.name {
            Vec::new()
          } else {
            vec![(Kind::TextDiffers, Some(id.clone()), Some("name".to_owned()), Some(Difference::Differs))]
          }
        }
        Change::Reweighed => {
          let (weight, difference) = match record.weight {
            Some((value, unit)) => ((value * 2.0, unit), Difference::Differs),
            None => ((1.0, "g"), Difference::OnlyInStore),
          };
          engine.evaluate(rewrite(record.name, Some(weight))).await.unwrap();
          vec![(Kind::QuantityDiffers, Some(id.clone()), Some("weight".to_owned()), Some(difference))]
        }
        Change::Rebranded { to } => {
          let mut stored = engine.store().get(&id).await.unwrap().unwrap();
          stored.brand_id = to.map(|index| BRAND_IDS[index].to_owned());
          engine.store().put(&stored).await.unwrap();
          match (record.brand_id, to) {
            (given, kept) if given == kept => Vec::new(),
            (Some(_), None) => vec![(Kind::BrandNotStored, Some(id.clone()), None, None)],
            _ => vec![(Kind::BrandDiffers, Some(id.clone()), None, None)],
          }
        }
        Change::Extra => {
          import_export(&engine, r#"{"id": "extra", "record": {"name": "live resin"}}"#).await;
          vec![(Kind::NotInExport, Some("extra".to_owned()), None, None)]
        }
        Change::Unlisted => {
          export_lines.remove(chosen);
          vec![(Kind::NotInExport, Some(id.clone()), None, None)]
        }
        Change::Rewritten => Vec::new(),
      }
    });

    let mut export = export_text(&export_lines);
    if matches!(change, Change::Rewritten) {
      let mut line = serde_json::json!({
        "record": {"name": format!("  {} \n", NAMES[record.name].replace(' ', " \t "))},
        "id": id,
      });
      if let Some(category) = record.category {
        line["record"]["category"] = format!("{}   ", CATEGORIES[category]).into();
      }
      match record.weight {
        Some((value, "g")) => line["record"]["weight"] = serde_json::json!({"min": value * 1000.0, "max": value * 1000.0, "unit": " MG"}),
        Some((value, _)) => line["record"]["weight"] = serde_json::json!({"value": value / 1000.0, "unit": "g"}),
        None => {}
      }
      if let Some(brand_id) = record.brand_id {
        line["brand_id"] = BRAND_IDS[brand_id].into();
      }
      export_lines[chosen] = record.clone();
      export = export_lines
        .iter()
        .enumerate()
        .map(|(index, other)| if index == chosen { line.to_string() + "\n" } else { other.json() + "\n" })
        .collect();
    }

    let (summary, findings) = runtime.block_on(audited(engine.store(), &export, WHOLE_STORE));
    prop_assert_eq!(told(&findings), expected.clone(), "{:?} of {:?}", change, record);
    // Every other record agrees. The chosen one agrees if it is still on both sides and nothing was found about
    // it: an extra record is about no record of the catalog.
    let chosen_agrees = match change {
      Change::Deleted | Change::Unlisted => false,
      Change::Extra => true,
      Change::Renamed { .. } | Change::Reweighed | Change::Rebranded { .. } | Change::Rewritten => expected.is_empty(),
    };
    prop_assert_eq!(summary.agree, records.len() - 1 + usize::from(chosen_agrees));
    prop_assert_eq!(summary.differences(), usize::from(!expected.is_empty()));
  }

  /// Whatever a store holds and whatever an export says, an audit counts everything it looks at once: each line,
  /// each point, and each record, which ends as agreeing, differing, missing on one side, or not compared. And
  /// it finds the same whatever the size of the pages it reads the store in.
  #[test]
  fn the_counts_of_an_audit_add_up(
    stored_lines in proptest::collection::vec(export_line(), 0..10),
    export_lines in proptest::collection::vec(export_line(), 0..10),
    other_layout in proptest::collection::btree_set(0..AUDIT_IDS.len(), 0..3),
    page_records in 1..4_usize,
    exported_at in proptest::option::of(prop_oneof![Just(0_u64), Just(u64::MAX)]),
  ) {
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let (layout, dimension) = store_layout(&schema(), &WordBagEmbedder::new("word-bag-v1"));
    let export = export_text(&export_lines);

    let (engine, ids_stored) = runtime.block_on(async {
      // What a collection that outlived a change of model looks like: some of its records under the old layout.
      let old = MemoryStore::new("another-model|category,description,name", dimension);
      for id in &other_layout {
        // Embedded by an engine of its own, so that what is put under the old layout is a whole record.
        let scratch = engine_on(empty_store(), None);
        let line = format!(r#"{{"id": "{}", "record": {{"name": "an older record"}}}}"#, AUDIT_IDS[*id]);
        import_export(&scratch, &line).await;
        let record = scratch.store().get(AUDIT_IDS[*id]).await.unwrap().unwrap();
        old.put(&record).await.unwrap();
      }
      let engine = engine_on(old.reopened(layout.clone(), dimension), None);
      import_export(&engine, &export_text(&stored_lines)).await;
      let ids: Vec<String> = engine
        .store()
        .scan(None, 1000, Read::Payload)
        .await
        .unwrap()
        .entries
        .into_iter()
        .map(|entry| match entry {
          Entry::Record(record) => record.id,
          Entry::OtherLayout { id, .. } => id,
          Entry::Unreadable { .. } => unreachable!("the in-memory store reads everything it holds"),
        })
        .collect();
      (engine, ids)
    });

    let options = audit::Options { page_records, exported_at };
    let (summary, findings) = runtime.block_on(audited(engine.store(), &export, options));
    let (whole, mut whole_findings) =
      runtime.block_on(audited(engine.store(), &export, audit::Options { page_records: 1000, exported_at }));
    prop_assert_eq!(&summary, &whole, "pages of {}", page_records);
    let mut paged_findings = findings.clone();
    let order = |finding: &Finding| (finding.line, finding.id.clone(), finding.kind, finding.field.clone(), finding.quantity.clone());
    paged_findings.sort_by_key(order);
    whole_findings.sort_by_key(order);
    prop_assert_eq!(paged_findings, whole_findings);

    prop_assert_eq!(summary.findings, findings.len());
    prop_assert_eq!(summary.lines, export_lines.len());
    prop_assert_eq!(summary.lines, summary.invalid_lines + summary.duplicate_ids + summary.export_records);
    prop_assert_eq!(summary.points, ids_stored.len());
    prop_assert_eq!(summary.points, summary.stored_records + summary.other_layout + summary.unreadable);
    prop_assert_eq!(summary.other_layout, other_layout.len());
    prop_assert_eq!(summary.other_layouts.values().sum::<usize>(), summary.other_layout);

    // Everything looked at: each record either side names, and each line that names none.
    let mut records: std::collections::BTreeSet<&str> = ids_stored.iter().map(String::as_str).collect();
    records.extend(export_lines.iter().filter(|line| line.flaw != Flaw::NotJson).map(|line| AUDIT_IDS[line.id]));
    let nameless = export_lines.iter().filter(|line| line.flaw == Flaw::NotJson).count();
    prop_assert_eq!(
      summary.agree + summary.differ + summary.not_stored + summary.not_in_export + summary.not_compared,
      records.len() + nameless,
      "{:?}", summary
    );
    for of_the_differing in [summary.brand_differs, summary.brand_not_stored, summary.text_differs, summary.quantity_differs] {
      prop_assert!(of_the_differing <= summary.differ);
    }
    // Everything stored was written after the year 1970 and before the end of time.
    let set_aside = match exported_at {
      Some(0) => summary.differ + summary.not_in_export,
      _ => 0,
    };
    prop_assert_eq!(summary.after_export, set_aside);
    prop_assert_eq!(
      summary.differences(),
      summary.differ + summary.not_stored + summary.not_in_export + summary.not_compared - set_aside
    );
  }
}

/// The record a request asks to be stored, if it asks for one.
fn asked_to_store(request: &Request) -> Option<RecordInput> {
  evaluate_input(request)
    .map(|input| input.record)
    .or_else(|| import_input(request).map(|input| input.record))
}

/// Runs `requests` and keeps, for each record that is stored at the end, the request content it was last written
/// from: what the caller's own database would say the record is.
async fn catalog_after(engine: &TestEngine, requests: &[Request]) -> Result<Vec<Option<RecordInput>>, TestCaseError> {
  let mut catalog: Vec<Option<RecordInput>> = vec![None; IDS.len()];
  for request in requests {
    let before = stored(engine).await;
    step(engine, request).await?;
    let after = stored(engine).await;
    for (index, (was, is)) in before.iter().zip(&after).enumerate() {
      match is {
        None => catalog[index] = None,
        Some(record) if was.as_ref() != Some(record) => {
          // The request that wrote it, under the brand the record ended up with.
          catalog[index] = asked_to_store(request).map(|content| RecordInput {
            brand_id: record.brand_id.clone(),
            ..content
          });
        }
        Some(_) => {}
      }
    }
  }
  Ok(catalog)
}

/// A cosine of six nines: one vector, as far as rounding lets two computations of it agree.
fn one_vector(left: &[f32], right: &[f32]) -> bool {
  cosine(left, right).is_ok_and(|cosine| cosine >= 0.999_999)
}

fn same_amounts(left: &similarity_core::Quantities, right: &similarity_core::Quantities) -> bool {
  left.len() == right.len()
    && left.iter().all(|(name, amount)| {
      similarity_core::quantity::compare(name, right.get(name), None, Some(amount))
        .is_some_and(|score| score.change == similarity_core::QuantityChange::Unchanged)
    })
}

proptest! {
  #![proptest_config(cases())]

  /// D8 against a restore: whatever was done to a store before a backup of its anchors was taken, and whatever
  /// was done to it afterwards, restoring the backup writes to a record only if that record has never been
  /// written since it was created, is its own anchor, and is what the backup was taken of. Everything else it
  /// leaves exactly as it was.
  #[test]
  fn a_restore_changes_only_a_record_with_no_history_that_is_what_was_backed_up(
    before in proptest::collection::vec(request(), 0..25),
    after in proptest::collection::vec(request(), 0..15),
    anchor_threshold in proptest::option::of(Just(0.5_f32)),
  ) {
    let engine = engine(anchor_threshold, false);
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    runtime.block_on(async {
      catalog_after(&engine, &before).await?;
      let backup: Vec<KeptAnchor> = stored(&engine).await.iter().flatten().map(KeptAnchor::from).collect();
      catalog_after(&engine, &after).await?;

      for kept in &backup {
        let everything = stored(&engine).await;
        let was = engine.store().get(&kept.id).await.unwrap();
        let outcome = sound_engine(engine.restore_anchor(kept, true).await)?;
        let is = engine.store().get(&kept.id).await.unwrap();
        for (other_was, other_is) in everything.iter().zip(&stored(&engine).await) {
          if other_was.as_ref().map(|record| &record.id) != Some(&kept.id) {
            prop_assert_eq!(other_was, other_is, "restoring `{}` changed another record", &kept.id);
          }
        }
        match (outcome, was, is) {
          (Restored::NotStored, None, None) => {}
          (Restored::Restored, Some(was), Some(is)) => {
            prop_assert_eq!(was.revision, 1, "it had been written since it was created");
            prop_assert!(one_vector(&was.anchor, &was.vectors.composite), "it had an anchor that was not itself");
            prop_assert!(same_amounts(&was.anchor_quantities, &was.quantities));
            prop_assert_eq!(&was.vectors.hashes, &kept.hashes, "it was not the text that was backed up");
            prop_assert!(same_amounts(&was.quantities, &kept.quantities));
            prop_assert_eq!(&was.brand_id, &kept.brand_id);
            // What changed is the anchor and the count of writes, and nothing the next revision is compared with.
            prop_assert!(one_vector(&is.anchor, &kept.anchor));
            prop_assert!(same_amounts(&is.anchor_quantities, &kept.anchor_quantities));
            prop_assert_eq!(is.revision, 2);
            prop_assert_eq!(
              StoredRecord { anchor: was.anchor.clone(), anchor_quantities: was.anchor_quantities.clone(), revision: 1, updated_at: was.updated_at, ..is },
              was
            );
          }
          (Restored::Already, Some(was), Some(is)) => {
            prop_assert!(one_vector(&was.anchor, &kept.anchor));
            prop_assert_eq!(was, is);
          }
          (Restored::OtherBrand | Restored::HasHistory | Restored::Changed, Some(was), Some(is)) => {
            prop_assert_eq!(was, is, "a record that may not be restored was written");
          }
          (outcome, was, is) => prop_assert!(false, "{:?} of {:?}, which became {:?}", outcome, was, is),
        }
      }
      Ok::<(), TestCaseError>(())
    })?;
  }

  /// What the backup is for. A store is lost with whatever history its records had. The catalog is imported
  /// again from the caller's database, as it stood, and the anchors are restored: every record then has the
  /// anchor it had, and restoring a second time finds nothing to do.
  #[test]
  fn a_store_that_is_lost_imported_again_and_restored_has_every_anchor_it_had(
    requests in proptest::collection::vec(request(), 1..40),
    anchor_threshold in proptest::option::of(Just(0.5_f32)),
  ) {
    let lost = engine(anchor_threshold, false);
    let again = engine(anchor_threshold, false);
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    runtime.block_on(async {
      let catalog = catalog_after(&lost, &requests).await?;
      let had = stored(&lost).await;

      for (content, had) in catalog.iter().zip(&had) {
        prop_assert_eq!(content.is_some(), had.is_some());
        let (Some(content), Some(had)) = (content, had) else { continue };
        let imported = sound_engine(again.import(vec![ImportInput { record: content.clone(), anchor: None }]).await)?;
        prop_assert_eq!(&imported, &[Ok(Imported::Created)], "{:?}", content);

        let kept = KeptAnchor::from(had);
        let outcome = sound_engine(again.restore_anchor(&kept, true).await)?;
        // A record that was its own anchor when the store was lost is its own anchor again, with nothing to do.
        let own = one_vector(&had.anchor, &had.vectors.composite) && same_amounts(&had.anchor_quantities, &had.quantities);
        prop_assert_eq!(outcome, if own { Restored::Already } else { Restored::Restored }, "{:?}", had);

        let has = again.store().get(&had.id).await.unwrap().unwrap();
        prop_assert!(one_vector(&has.anchor, &had.anchor), "the anchor of `{}`", &had.id);
        prop_assert!(same_amounts(&has.anchor_quantities, &had.anchor_quantities));
        prop_assert!(one_vector(&has.vectors.composite, &had.vectors.composite));
        prop_assert_eq!((&has.vectors.hashes, &has.brand_id), (&had.vectors.hashes, &had.brand_id));
        prop_assert_eq!(sound_engine(again.restore_anchor(&kept, true).await)?, Restored::Already);
        prop_assert_eq!(again.store().get(&had.id).await.unwrap().unwrap(), has);
      }
      Ok::<(), TestCaseError>(())
    })?;
  }
}
