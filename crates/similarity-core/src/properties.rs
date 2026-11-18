//! Properties of the scoring code, checked on generated inputs.
//!
//! The tests beside each module are examples: this record, that edit, the number expected. These state what must
//! hold for *every* input and let the machine look for one that breaks it. The rules they cover are the ones a
//! verdict rests on: that a composite's cosine is the weighted mean it is documented to be (D5), that the anchor
//! never loses a value it holds (D8), that judging again equals judging afresh (D17), and that a quantity's
//! factor means the same thing whichever way round, and in whatever unit, it is asked (D16), and that a long text
//! is tokenized as if all of it had been read (D21).
//!
//! Each property runs a modest number of cases so that the fast suite stays fast. A failure is shrunk to a small
//! input and saved under `proptest-regressions/`, which is checked in: the case is then replayed first on every
//! run, on every machine.

use std::collections::BTreeMap;

use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};
use tokenizers::normalizers::BertNormalizer;

use crate::embed::EmbedderOptions;
use crate::matching::{self, Candidate};
use crate::quantity::{self, Quantities, Quantity, QuantityChange, first_values};
use crate::record::testing::fake_embedding;
use crate::record::{PreparedRecord, RecordVectors, normalize_text, text_hash};
use crate::schema::{FieldSpec, QuantitySpec, Schema};
use crate::score::{self, FieldChange, Policy, Reason};
use crate::separation::area_under_curve;
use crate::testing::{WORDS, tiny_embedder, tiny_tokenizer};
use crate::tokenize::TextTokenizer;
use crate::vector::{cosine, norm, normalize};

/// The width of one generated field vector. Small, so that a shrunk failure is readable.
const DIM: usize = 4;

/// 64 cases a property, unless `PROPTEST_CASES` asks for another number: `PROPTEST_CASES=5000 just test
/// similarity-core` is the thorough run.
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

/// A unit vector, which is what the encoder hands the scoring code for every field: `Schema::compose` packs unit
/// vectors and says so, and the composite identity only holds for them.
fn direction() -> impl Strategy<Value = Vec<f32>> {
  proptest::collection::vec(-1.0_f32..1.0, DIM)
    // Not so short that normalizing it amplifies rounding into the comparison.
    .prop_filter("needs a direction", |vector| norm(vector) > 0.1)
    .prop_map(|mut vector| {
      assert!(normalize(&mut vector));
      vector
    })
}

/// One generated field: its weight, its floor, and its vector on each side of an edit (absent where `None`).
#[derive(Debug, Clone)]
struct Field {
  weight: f32,
  floor: Option<f32>,
  old: Option<Vec<f32>>,
  new: Option<Vec<f32>>,
  original: Option<Vec<f32>>,
}

fn field() -> impl Strategy<Value = Field> {
  (
    0.1_f32..10.0,
    proptest::option::of(-1.0_f32..1.0),
    proptest::option::weighted(0.8, direction()),
    proptest::option::weighted(0.8, direction()),
    proptest::option::weighted(0.6, direction()),
  )
    .prop_map(|(weight, floor, old, new, original)| Field {
      weight,
      floor,
      old,
      new,
      original,
    })
}

/// Two to four fields of which every side (old, new, original) has at least one, since a record with no content
/// cannot be embedded and is refused long before scoring.
fn fields() -> impl Strategy<Value = Vec<Field>> {
  proptest::collection::vec(field(), 2..=4).prop_filter("every version needs some content", |fields| {
    fields.iter().any(|field| field.old.is_some())
      && fields.iter().any(|field| field.new.is_some())
      && fields.iter().any(|field| field.original.is_some())
  })
}

fn schema_of(fields: &[Field]) -> Schema {
  let specs = fields
    .iter()
    .enumerate()
    .map(|(index, field)| FieldSpec {
      name: format!("f{index}"),
      weight: field.weight,
      required: false,
      reject_below: field.floor,
    })
    .collect();
  Schema::new(specs).unwrap()
}

/// The stored form of one side of the generated fields. Every present field gets a text fingerprint unique to
/// the side, so that no field is taken for unchanged.
fn vectors(schema: &Schema, fields: &[Field], side: &str, pick: impl Fn(&Field) -> Option<&Vec<f32>>) -> RecordVectors {
  let blocks: Vec<Option<&[f32]>> = fields.iter().map(|field| pick(field).map(Vec::as_slice)).collect();
  let hashes: BTreeMap<String, String> = fields
    .iter()
    .enumerate()
    .filter(|(_, field)| pick(field).is_some())
    .map(|(index, _)| (format!("f{index}"), format!("{side}{index}")))
    .collect();
  RecordVectors {
    composite: schema.compose(DIM, blocks).unwrap(),
    hashes,
  }
}

/// The weighted mean of the cosines of the fields both `left` and `right` pick, computed the plain way.
fn expected_mean(
  fields: &[Field],
  left: impl Fn(&Field) -> Option<&Vec<f32>>,
  right: impl Fn(&Field) -> Option<&Vec<f32>>,
) -> Option<f64> {
  let (mut weighted, mut total) = (0.0_f64, 0.0_f64);
  for field in fields {
    if let (Some(a), Some(b)) = (left(field), right(field)) {
      weighted += f64::from(field.weight) * f64::from(cosine(a, b).unwrap());
      total += f64::from(field.weight);
    }
  }
  (total > 0.0).then(|| weighted / total)
}

proptest! {
  #![proptest_config(cases())]

  /// D5: when both records have every field, the cosine of their composites is the weighted mean of the field
  /// cosines. This identity is why the score Qdrant computes in a search is the score a verdict uses.
  #[test]
  fn the_cosine_of_two_full_composites_is_the_weighted_mean_of_their_fields(
    weights in proptest::collection::vec(0.1_f32..10.0, 2..=4),
    seed in proptest::collection::vec((direction(), direction()), 4),
  ) {
    let fields: Vec<Field> = weights
      .iter()
      .zip(&seed)
      .map(|(weight, (old, new))| Field {
        weight: *weight,
        floor: None,
        old: Some(old.clone()),
        new: Some(new.clone()),
        original: None,
      })
      .collect();
    let schema = schema_of(&fields);
    let old = vectors(&schema, &fields, "old", |field| field.old.as_ref());
    let new = vectors(&schema, &fields, "new", |field| field.new.as_ref());
    let expected = expected_mean(&fields, |field| field.old.as_ref(), |field| field.new.as_ref()).unwrap();
    let found = f64::from(cosine(&old.composite, &new.composite).unwrap());
    prop_assert!((found - expected).abs() < 1e-4, "{found} against {expected}");
  }

  /// A comparison reports exactly the fields either side has, names what happened to each from which sides have
  /// it, scores the ones that can be scored, and averages those by weight. Every number is a cosine.
  #[test]
  fn a_comparison_classifies_and_scores_every_field_from_which_sides_have_it(fields in fields()) {
    let schema = schema_of(&fields);
    let old = vectors(&schema, &fields, "old", |field| field.old.as_ref());
    let new = vectors(&schema, &fields, "new", |field| field.new.as_ref());
    let original = vectors(&schema, &fields, "original", |field| field.original.as_ref());
    let comparison = score::compare(&schema, DIM, &old, Some(&original.composite), &new).unwrap();

    prop_assert!(comparison.similarity.is_finite() && (-1.0..=1.0).contains(&comparison.similarity));
    let mut reported = comparison.fields.iter();
    for (index, field) in fields.iter().enumerate() {
      let expected = match (&field.old, &field.new, &field.original) {
        (None, None, _) => continue,
        (Some(_), Some(_), _) => FieldChange::Modified,
        (Some(_), None, _) => FieldChange::Removed,
        (None, Some(_), Some(_)) => FieldChange::Restored,
        (None, Some(_), None) => FieldChange::Added,
      };
      let score = reported.next().expect("a score for every field either side has");
      prop_assert_eq!(&score.field, &format!("f{index}"));
      prop_assert_eq!(score.change, expected);
      let reference = match expected {
        FieldChange::Modified => field.old.as_ref(),
        FieldChange::Restored => field.original.as_ref(),
        _ => None,
      };
      match (score.similarity, reference.zip(field.new.as_ref())) {
        (Some(found), Some((reference, new))) => {
          prop_assert!((-1.0..=1.0).contains(&found));
          let wanted = cosine(reference, new).unwrap();
          prop_assert!((found - wanted).abs() < 1e-4, "f{}: {} against {}", index, found, wanted);
          prop_assert_eq!(score.below_floor, field.floor.is_some_and(|floor| found < floor));
        }
        (None, None) => prop_assert!(!score.below_floor),
        (found, wanted) => prop_assert!(false, "f{}: similarity {:?} where {:?} was expected", index, found, wanted.is_some()),
      }
    }
    prop_assert!(reported.next().is_none(), "no score for a field neither side has");

    // The overall score is the weighted mean of exactly the scores it reported.
    let (mut weighted, mut total) = (0.0_f64, 0.0_f64);
    for score in &comparison.fields {
      if let Some(similarity) = score.similarity {
        weighted += f64::from(score.weight) * f64::from(similarity);
        total += f64::from(score.weight);
      }
    }
    let expected = if total > 0.0 { weighted / total } else { 0.0 };
    prop_assert!((f64::from(comparison.similarity) - expected).abs() < 1e-5);
  }

  /// D8: an accepted write gives the anchor the first value of a field it never had, and never replaces a value
  /// it holds.
  #[test]
  fn extending_the_anchor_keeps_every_value_it_holds_and_takes_only_what_it_lacks(fields in fields()) {
    let schema = schema_of(&fields);
    let anchor = vectors(&schema, &fields, "original", |field| field.original.as_ref()).composite;
    let candidate = vectors(&schema, &fields, "new", |field| field.new.as_ref()).composite;
    let extended = score::extend_anchor(&schema, DIM, &anchor, &candidate).unwrap();

    let blocks = schema.blocks(DIM, &extended).unwrap();
    for (block, field) in blocks.iter().zip(&fields) {
      match (field.original.as_ref().or(field.new.as_ref()), block) {
        (Some(wanted), Some(found)) => {
          // The anchor's own value where it had one, the candidate's only where it did not.
          let agreement = cosine(wanted, found).unwrap();
          prop_assert!(agreement > 0.9999, "{agreement}");
        }
        (None, None) => {}
        (wanted, found) => prop_assert!(false, "block {:?} where {:?} was expected", found.is_some(), wanted.is_some()),
      }
    }
    // And doing it again changes nothing: there is nothing left to take.
    let again = score::extend_anchor(&schema, DIM, &extended, &candidate).unwrap();
    prop_assert_eq!(again, extended);
  }

  /// D17: weights, floors, and thresholds can be changed on a comparison that was already measured, and the
  /// result is what measuring again under them would have given.
  #[test]
  fn judging_again_under_another_configuration_equals_evaluating_afresh(
    fields in fields(),
    other in proptest::collection::vec((0.1_f32..10.0, proptest::option::of(-1.0_f32..1.0)), 4),
    thresholds in (-1.0_f32..1.0, proptest::option::of(-1.0_f32..1.0), -1.0_f32..1.0, proptest::option::of(-1.0_f32..1.0)),
  ) {
    let first = schema_of(&fields);
    let reweighted: Vec<Field> = fields
      .iter()
      .zip(&other)
      .map(|(field, (weight, floor))| Field { weight: *weight, floor: *floor, ..field.clone() })
      .collect();
    let second = schema_of(&reweighted);
    let old = vectors(&first, &fields, "old", |field| field.old.as_ref());
    let new = vectors(&first, &fields, "new", |field| field.new.as_ref());
    let anchor = vectors(&first, &fields, "original", |field| field.original.as_ref()).composite;
    let lenient = Policy::new(thresholds.0, thresholds.1).unwrap();
    let strict = Policy::new(thresholds.2, thresholds.3).unwrap();

    let mut judged_again = score::evaluate(&first, &lenient, DIM, &old, Some(&anchor), &new).unwrap();
    judged_again.rejudge(&second, &strict);

    // Composites written under the first weights are still what the second schema reads: weights never change
    // a stored comparison. The anchor similarity is the one thing judging again cannot recompute (it has only
    // the number, not the vectors), so it is taken as measured.
    let mut afresh = score::evaluate(&second, &strict, DIM, &old, Some(&anchor), &new).unwrap();
    afresh.anchor_similarity = judged_again.anchor_similarity;
    afresh.reasons = score::judge(&second, &strict, &afresh.comparison, afresh.anchor_similarity);

    prop_assert!((judged_again.comparison.similarity - afresh.comparison.similarity).abs() < 1e-5);
    let codes = |reasons: &[Reason]| reasons.iter().map(|reason| (reason.code(), reason.subject().map(str::to_owned))).collect::<Vec<_>>();
    // A similarity within rounding of a threshold or floor may fall either side of it; anywhere else the
    // verdicts must be identical.
    let near_a_limit = judged_again.comparison.fields.iter().zip(&reweighted).any(|(score, field)| {
      matches!((score.similarity, field.floor), (Some(found), Some(floor)) if (found - floor).abs() < 1e-4)
    }) || (judged_again.comparison.similarity - strict.threshold).abs() < 1e-4;
    if !near_a_limit {
      prop_assert_eq!(codes(&judged_again.reasons), codes(&afresh.reasons));
    }
  }

  /// Whitespace normalization is idempotent, and padding or re-wrapping a text changes neither it nor its hash.
  #[test]
  fn normalizing_text_twice_is_normalizing_it_once(text in "[ \\t\\n\\r]{0,3}([a-zA-Z0-9éß%.,-]{0,6}[ \\t\\n\\r]{0,3}){0,6}") {
    let once = normalize_text(&text);
    match &once {
      Some(normalized) => {
        prop_assert_eq!(normalize_text(normalized), once.clone());
        prop_assert!(!normalized.starts_with(' ') && !normalized.ends_with(' ') && !normalized.contains("  "));
        let padded = format!(" \n{}\t ", text.replace(' ', "\n  "));
        let again = normalize_text(&padded);
        prop_assert_eq!(again.as_ref(), Some(normalized));
        prop_assert_eq!(text_hash(&again.unwrap()), text_hash(normalized));
      }
      None => prop_assert!(text.trim().is_empty()),
    }
  }
}

/// A unit the table converts, with its dimension's other spellings to rewrite into.
fn known_unit() -> impl Strategy<Value = (&'static str, &'static str)> {
  prop_oneof![
    Just(("mg", "g")),
    Just(("g", "mg")),
    Just(("g", "oz")),
    Just(("kg", "lb")),
    Just(("mcg", "mg")),
    Just(("ml", "l")),
    Just(("fl oz", "ml")),
    Just(("cl", "ml")),
    Just(("%", "percent")),
  ]
}

/// An amount as a record might hold it: usually ordinary, sometimes zero, sometimes tiny or huge.
fn amount() -> impl Strategy<Value = f64> {
  prop_oneof![
    6 => 0.001_f64..10_000.0,
    1 => Just(0.0),
    1 => 1e-9_f64..1e-3,
    1 => 1e4_f64..1e12,
  ]
}

fn range() -> impl Strategy<Value = (f64, f64)> {
  (amount(), amount()).prop_map(|(a, b)| if a <= b { (a, b) } else { (b, a) })
}

/// `quantity` written in `unit`, which must be of the same dimension.
fn rewritten(quantity: &Quantity, unit: &str) -> Quantity {
  let (_, from) = quantity::unit(&quantity.unit);
  let (_, to) = quantity::unit(unit);
  Quantity::range(quantity.min * from / to, quantity.max * from / to, unit)
}

proptest! {
  #![proptest_config(cases())]

  /// D16: a factor is the larger over the smaller, so it reads the same in both directions and is never below
  /// 1; and whether two quantities can be compared at all does not depend on the direction either.
  #[test]
  fn a_factor_is_the_same_in_both_directions_and_at_least_one(
    (unit, other) in known_unit(), before in range(), after in range(), same_unit in any::<bool>(),
  ) {
    let old = Quantity::range(before.0, before.1, unit);
    let new = Quantity::range(after.0, after.1, if same_unit { unit } else { other });
    prop_assert!(old.check().is_ok() && new.check().is_ok());
    let forward = quantity::compare("q", Some(&old), None, Some(&new)).unwrap();
    let backward = quantity::compare("q", Some(&new), None, Some(&old)).unwrap();

    prop_assert_eq!(forward.change, backward.change);
    prop_assert_eq!(forward.zero_crossing, backward.zero_crossing);
    prop_assert!(!forward.incomparable, "two units of one dimension");
    match (forward.factor, backward.factor) {
      (Some(there), Some(back)) => {
        prop_assert!(there.is_finite() && there >= 1.0, "{}", there);
        prop_assert!((there - back).abs() <= 1e-9 * there, "{} against {}", there, back);
        prop_assert_eq!(forward.change == QuantityChange::Unchanged, there <= 1.0);
      }
      (None, None) => prop_assert!(forward.zero_crossing, "no factor only when one end moved to or from zero"),
      (there, back) => prop_assert!(false, "{:?} one way and {:?} the other", there, back),
    }
  }

  /// D16: the same amount in another unit is the same amount, and rewriting both sides of a change in another
  /// unit does not change how big the change is.
  #[test]
  fn rewriting_a_quantity_in_another_unit_changes_nothing(
    (unit, other) in known_unit(), before in range(), after in range(),
  ) {
    let old = Quantity::range(before.0, before.1, unit);
    let new = Quantity::range(after.0, after.1, unit);

    let same = quantity::compare("q", Some(&old), None, Some(&rewritten(&old, other))).unwrap();
    prop_assert_eq!(same.change, QuantityChange::Unchanged);
    prop_assert_eq!(same.factor, Some(1.0));

    let plain = quantity::compare("q", Some(&old), None, Some(&new)).unwrap();
    let converted = quantity::compare("q", Some(&rewritten(&old, other)), None, Some(&rewritten(&new, other))).unwrap();
    prop_assert_eq!(plain.zero_crossing, converted.zero_crossing);
    match (plain.factor, converted.factor) {
      (Some(a), Some(b)) => prop_assert!((a - b).abs() <= 1e-6 * a, "{} against {}", a, b),
      (a, b) => prop_assert_eq!(a, b),
    }
  }

  /// D16: quantities are judged beside the text and never inside it. Judging them changes no similarity and
  /// removes no reason; with no limit configured it adds none; and a limit only ever adds.
  #[test]
  fn judging_quantities_never_touches_the_text_verdict(
    fields in fields(),
    (unit, other) in known_unit(), before in proptest::option::of(range()), after in proptest::option::of(range()),
    original in proptest::option::of(range()), switch_unit in any::<bool>(),
    limit in proptest::option::of(1.0_f64..50.0), reject_incomparable in any::<bool>(), on_measure in any::<bool>(),
    threshold in -1.0_f32..1.0,
  ) {
    let named = |limit: Option<f64>, reject_incomparable: bool| QuantitySpec {
      name: "q".to_owned(),
      reject_above_factor: limit,
      reject_incomparable,
    };
    let quantities = |value: Option<(f64, f64)>, unit: &str| -> Quantities {
      value.map(|(min, max)| ("q".to_owned(), Quantity::range(min, max, unit))).into_iter().collect()
    };
    let (old_q, new_q) = (quantities(before, unit), quantities(after, if switch_unit { "each" } else { other }));
    let anchor_q = quantities(original, unit);

    let text_only = schema_of(&fields);
    let old = vectors(&text_only, &fields, "old", |field| field.old.as_ref());
    let new = vectors(&text_only, &fields, "new", |field| field.new.as_ref());
    let mut policy = Policy::new(threshold, None).unwrap();
    let text = score::evaluate(&text_only, &policy, DIM, &old, None, &new).unwrap();

    // Reported only: nothing is added.
    let reporting = schema_of(&fields).with_quantities(vec![named(None, false)]).unwrap();
    let mut reported = text.clone();
    reported.judge_quantities(&reporting, &policy, &old_q, &anchor_q, &new_q);
    prop_assert_eq!(&reported.comparison, &text.comparison);
    prop_assert_eq!(&reported.reasons, &text.reasons);
    prop_assert!(reported.quantities.iter().all(|score| !score.beyond_limit));
    prop_assert_eq!(reported.quantities.len(), usize::from(before.is_some() || after.is_some()));

    // Limited: the text verdict is still there, untouched, at the front.
    policy.reject_on_measure_change = on_measure;
    let limiting = schema_of(&fields).with_quantities(vec![named(limit, reject_incomparable)]).unwrap();
    let mut limited = text.clone();
    limited.judge_quantities(&limiting, &policy, &old_q, &anchor_q, &new_q);
    prop_assert_eq!(&limited.comparison, &text.comparison);
    prop_assert!(limited.reasons.starts_with(&text.reasons));
    let added = &limited.reasons[text.reasons.len()..];
    prop_assert!(added.iter().all(|reason| reason.code().starts_with("quantity_") || reason.code() == "measure_changed"));
    // What was measured does not depend on what it is held to.
    prop_assert_eq!(limited.measure_changed, reported.measure_changed);
    for (with_limit, without) in limited.quantities.iter().zip(&reported.quantities) {
      prop_assert_eq!((with_limit.change, with_limit.factor), (without.change, without.factor));
      prop_assert_eq!(with_limit.beyond_limit, added.iter().any(|reason| reason.subject() == Some("q")));
    }
  }

  /// D16: the anchor keeps the first value of every quantity, like the first value of every field.
  #[test]
  fn the_first_values_of_quantities_are_never_replaced(
    anchor in proptest::collection::btree_map("[a-d]", amount(), 0..4),
    candidate in proptest::collection::btree_map("[a-f]", amount(), 0..5),
  ) {
    let as_quantities = |values: &BTreeMap<String, f64>, unit: &str| -> Quantities {
      values.iter().map(|(name, value)| (name.clone(), Quantity::single(*value, unit))).collect()
    };
    let (anchor, candidate) = (as_quantities(&anchor, "g"), as_quantities(&candidate, "mg"));
    let extended = first_values(&anchor, &candidate);

    for (name, held) in &anchor {
      prop_assert_eq!(extended.get(name), Some(held));
    }
    for (name, offered) in &candidate {
      prop_assert_eq!(extended.get(name), Some(anchor.get(name).unwrap_or(offered)));
    }
    prop_assert!(extended.keys().all(|name| anchor.contains_key(name) || candidate.contains_key(name)));
    prop_assert_eq!(first_values(&extended, &candidate), extended);
  }

  /// A quantity survives the trip through JSON exactly, in the spelling it is stored in.
  #[test]
  fn a_quantity_reads_back_from_json_as_it_was_written((min, max) in range(), unit in "[a-z%/ ]{1,6}") {
    let quantity = Quantity::range(min, max, unit);
    let json = serde_json::to_string(&quantity).unwrap();
    prop_assert_eq!(serde_json::from_str::<Quantity>(&json).unwrap(), quantity);
  }
}

/// D9: a text embeds to the same vector alone and inside a merged, length-bucketed batch. Requests are batched by
/// backlog, so which other texts share a forward pass with a field is an accident of timing and must not move a
/// score.
#[test]
fn a_text_embeds_the_same_alone_and_in_any_batch() {
  let embedder = tiny_embedder(&EmbedderOptions {
    max_tokens: 16,
    batch_size: 4,
    output_dim: None,
    threads: None,
  });
  // Sentences of one to seven known words, so that a batch mixes lengths and has to be padded.
  let sentence =
    proptest::collection::vec(proptest::sample::select(WORDS.to_vec()), 1..=7).prop_map(|words| words.join(" "));
  let batch = proptest::collection::vec(sentence, 1..=9);

  TestRunner::new(cases())
    .run(&batch, |texts| {
      let together = embedder.embed(&texts).unwrap();
      prop_assert_eq!(together.len(), texts.len());
      for (text, in_batch) in texts.iter().zip(&together) {
        let alone = embedder.embed(std::slice::from_ref(text)).unwrap();
        let agreement = cosine(&alone[0], in_batch).unwrap();
        prop_assert!(agreement > 0.9999, "`{}` alone and in the batch: {}", text, agreement);
        prop_assert!((norm(in_batch) - 1.0).abs() < 1e-4, "every vector is a unit vector");
      }
      Ok(())
    })
    .unwrap();
}

/// A stretch of a field's text: a word, or a run of one thing repeated, long enough to outgrow the part of a text
/// that is tokenized first. The runs are what padding is made of: characters BERT's normalizer deletes
/// (zero-width space, word joiner, form feed), ones it strips (a combining accent), an unbroken word, and every
/// kind of gap.
fn stretch() -> impl Strategy<Value = String> {
  let repeated = vec![
    "\u{200b}", "\u{2060}", "\u{c}", "\u{301}", "\u{a0}", "a", "é", "ブ", ".", " ", "\t", "\n",
  ];
  prop_oneof![
    proptest::sample::select(WORDS.to_vec()).prop_map(str::to_owned),
    Just("unheard".to_owned()),
    proptest::sample::select(vec![" ", "\t", "\n", "\r", "\u{c}", "\u{a0}", ""]).prop_map(str::to_owned),
    (proptest::sample::select(repeated), 0..400_usize).prop_map(|(unit, times)| unit.repeat(times)),
  ]
}

proptest! {
  #![proptest_config(cases())]

  /// D21: only a head of a long text is handed to the tokenizer, and the tokens must be the ones the whole text
  /// gives, whatever the text is made of. The first version of the cut counted bytes, and padding a field with
  /// characters that come to no tokens pushed its words out of what was read.
  #[test]
  fn a_text_is_tokenized_as_if_all_of_it_had_been_read(
    stretches in proptest::collection::vec(stretch(), 0..12),
    max_tokens in 3..9_usize,
    bert in any::<bool>(),
  ) {
    let mut tokenizer = tiny_tokenizer();
    if bert {
      let _ = tokenizer.with_normalizer(Some(BertNormalizer::default()));
    }
    let tokenizer = TextTokenizer::new(tokenizer, max_tokens).unwrap();
    let text = stretches.concat();
    prop_assert_eq!(tokenizer.encode(&text).unwrap(), tokenizer.encode_whole(&text).unwrap());
  }
}

/// A handful of scores on a coarse grid, so that ties are common.
fn scores() -> impl Strategy<Value = Vec<f32>> {
  proptest::collection::vec((-4_i8..=20).prop_map(|step| f32::from(step) / 20.0), 1..12)
}

proptest! {
  #![proptest_config(cases())]

  /// The area under the curve is the share of (edit, recycle) pairings the edit wins, a tie counting half:
  /// counted here one pairing at a time, which is slow and obviously right.
  #[test]
  fn the_area_under_the_curve_is_the_share_of_pairings_an_edit_wins(edits in scores(), recycles in scores()) {
    let mut wins = 0.0_f64;
    for edit in &edits {
      for recycle in &recycles {
        wins += if edit > recycle { 1.0 } else if edit == recycle { 0.5 } else { 0.0 };
      }
    }
    #[expect(clippy::cast_precision_loss, reason = "a few dozen pairings")]
    let expected = wins / (edits.len() * recycles.len()) as f64;
    let area = area_under_curve(&edits, &recycles).unwrap();
    prop_assert!((area - expected).abs() < 1e-12, "{} against {}", area, expected);
    prop_assert!((0.0..=1.0).contains(&area));

    // Asked the other way round, it is the complement.
    let reversed = area_under_curve(&recycles, &edits).unwrap();
    prop_assert!((area + reversed - 1.0).abs() < 1e-12);

    // It depends on the order of the scores and nothing else, which is why it needs no threshold: any rescaling
    // that keeps the order keeps the area.
    let rescale = |values: &[f32]| values.iter().map(|value| value * 3.0 - 0.25).collect::<Vec<f32>>();
    let rescaled = area_under_curve(&rescale(&edits), &rescale(&recycles)).unwrap();
    prop_assert!((area - rescaled).abs() < 1e-12);
  }
}

/// A catalog for matching: the weights of two to four fields, a menu item, and some stored records, each a value
/// or none for every field, with at least one value.
type Catalog = (Vec<f32>, Vec<Option<Vec<f32>>>, Vec<Vec<Option<Vec<f32>>>>);

fn catalog() -> impl Strategy<Value = Catalog> {
  (2_usize..=4).prop_flat_map(|count| {
    let filled = move || {
      proptest::collection::vec(proptest::option::weighted(0.7, direction()), count)
        .prop_filter("a record needs some content", |blocks| {
          blocks.iter().any(Option::is_some)
        })
    };
    (
      proptest::collection::vec(0.1_f32..10.0, count),
      filled(),
      proptest::collection::vec(filled(), 1..6),
    )
  })
}

fn weighted_schema(weights: &[f32]) -> Schema {
  let specs = weights
    .iter()
    .enumerate()
    .map(|(index, weight)| FieldSpec {
      name: format!("f{index}"),
      weight: *weight,
      required: false,
      reject_below: Some(0.5),
    })
    .collect();
  Schema::new(specs).unwrap()
}

/// Stored vectors for `blocks`, with text fingerprints unique to `side` so that nothing is taken for unchanged.
fn stored(schema: &Schema, blocks: &[Option<Vec<f32>>], side: &str) -> RecordVectors {
  let hashes = blocks
    .iter()
    .enumerate()
    .filter(|(_, block)| block.is_some())
    .map(|(index, _)| (format!("f{index}"), format!("{side}{index}")))
    .collect();
  RecordVectors {
    composite: schema.compose(DIM, blocks.iter().map(Option::as_deref)).unwrap(),
    hashes,
  }
}

proptest! {
  #![proptest_config(cases())]

  /// Why a match is scored again after the search (D5, D32): the cosine of two composites is the weighted mean
  /// over the fields both have, times a factor that depends only on which fields each side filled in.
  #[test]
  fn the_search_score_is_the_match_score_scaled_by_the_field_overlap((weights, menu_item, records) in catalog()) {
    let schema = weighted_schema(&weights);
    let menu_item_vectors = stored(&schema, &menu_item, "menu_item");
    let none = Quantities::new();
    for (index, record) in records.iter().enumerate() {
      let vectors = stored(&schema, record, &format!("record{index}"));
      let search = f64::from(cosine(&menu_item_vectors.composite, &vectors.composite).unwrap());
      let candidate = Candidate { id: "r", brand_id: None, search_similarity: 0.0, vectors: &vectors, quantities: &none };
      let ranked = matching::rank(&schema, DIM, &menu_item_vectors, &none, [candidate], 1).unwrap();
      let exact = f64::from(ranked[0].comparison.similarity);

      let weight_of = |blocks: &[Option<Vec<f32>>], other: &[Option<Vec<f32>>]| -> f64 {
        weights
          .iter()
          .zip(blocks.iter().zip(other))
          .filter(|(_, (block, other))| block.is_some() && other.is_some())
          .map(|(weight, _)| f64::from(*weight))
          .sum()
      };
      let (both, of_menu_item, of_record) = (
        weight_of(&menu_item, record),
        weight_of(&menu_item, &menu_item),
        weight_of(record, record),
      );
      let factor = both / (of_menu_item * of_record).sqrt();
      prop_assert!((search - exact * factor).abs() < 1e-4, "search {search}, exact {exact}, factor {factor}");
      prop_assert!(factor <= 1.0 + 1e-9);
      prop_assert!(ranked[0].comparison.fields.iter().all(|field| !field.below_floor));
    }
  }

  /// The ranking is the plain weighted mean over shared fields, best first, whatever order the candidates came
  /// in; a limit keeps the front of it; and a menu item every one of whose values a record holds scores one.
  #[test]
  fn a_ranking_is_the_exact_order_whatever_order_the_candidates_came_in(
    (weights, menu_item, records) in catalog(),
    limit in 0_usize..7,
  ) {
    let schema = weighted_schema(&weights);
    let menu_item_vectors = stored(&schema, &menu_item, "menu_item");
    let none = Quantities::new();
    let ids: Vec<String> = (0..records.len()).map(|index| format!("r{index}")).collect();
    let vectors: Vec<RecordVectors> = records
      .iter()
      .enumerate()
      .map(|(index, record)| stored(&schema, record, &format!("record{index}")))
      .collect();
    let candidates = || {
      ids.iter().zip(&vectors).map(|(id, vectors)| Candidate {
        id,
        brand_id: None,
        search_similarity: 0.0,
        vectors,
        quantities: &none,
      })
    };
    let all = matching::rank(&schema, DIM, &menu_item_vectors, &none, candidates(), usize::MAX).unwrap();
    prop_assert_eq!(all.len(), records.len());

    // Each score is the mean computed the plain way.
    for ranked in &all {
      let index: usize = ranked.id[1..].parse().unwrap();
      let (mut weighted, mut total) = (0.0_f64, 0.0_f64);
      for (weight, (ours, theirs)) in weights.iter().zip(menu_item.iter().zip(&records[index])) {
        if let (Some(ours), Some(theirs)) = (ours, theirs) {
          weighted += f64::from(*weight) * f64::from(cosine(theirs, ours).unwrap());
          total += f64::from(*weight);
        }
      }
      let expected = if total > 0.0 { weighted / total } else { 0.0 };
      prop_assert!((f64::from(ranked.comparison.similarity) - expected).abs() < 1e-5);
    }
    // Best first, ties by id.
    for pair in all.windows(2) {
      let (first, second) = (pair[0].comparison.similarity, pair[1].comparison.similarity);
      prop_assert!(first > second || (first == second && pair[0].id < pair[1].id));
    }
    // The same from any order, and a limit keeps the front.
    let reversed = matching::rank(&schema, DIM, &menu_item_vectors, &none, candidates().rev(), usize::MAX).unwrap();
    prop_assert_eq!(&reversed, &all);
    let limited = matching::rank(&schema, DIM, &menu_item_vectors, &none, candidates(), limit).unwrap();
    prop_assert_eq!(&limited[..], &all[..limit.min(all.len())]);

    // A record that holds every value the menu item has, and more, is a perfect match for it.
    let superset: Vec<Option<Vec<f32>>> = menu_item
      .iter()
      .zip(&records[0])
      .map(|(ours, theirs)| ours.clone().or_else(|| theirs.clone()))
      .collect();
    let superset = stored(&schema, &superset, "superset");
    let own = Candidate { id: "own", brand_id: None, search_similarity: 0.0, vectors: &superset, quantities: &none };
    let ranked = matching::rank(&schema, DIM, &menu_item_vectors, &none, [own], 1).unwrap();
    prop_assert!((ranked[0].comparison.similarity - 1.0).abs() < 1e-5);
  }
}

/// A field's text as a caller might send it: nothing, only whitespace, or a few words with stray whitespace
/// around and between them.
fn field_text() -> impl Strategy<Value = Option<String>> {
  prop_oneof![
    Just(None),
    proptest::sample::select(vec!["", " ", "\n\t"]).prop_map(|blank| Some(blank.to_owned())),
    spaced_words().prop_map(Some),
  ]
}

/// A few words with stray whitespace around and between them: a field that has a value.
fn spaced_words() -> impl Strategy<Value = String> {
  (
    proptest::sample::select(vec!["", " ", "\n", " \t "]),
    proptest::collection::vec(proptest::sample::select(WORDS.to_vec()), 1..4),
    proptest::sample::select(vec![" ", "  ", "\n"]),
  )
    .prop_map(|(around, words, between)| format!("{around}{}{around}", words.join(between)))
}

/// The texts of a record's three fields, at least one of which has a value: a record with none is refused before
/// it has fingerprints, and a case built of one would assert nothing.
fn record_texts() -> impl Strategy<Value = Vec<Option<String>>> {
  (proptest::collection::vec(field_text(), 3), 0..3_usize, spaced_words()).prop_map(|(mut texts, at, words)| {
    if let Some(text) = texts.get_mut(at) {
      *text = Some(words);
    }
    texts
  })
}

proptest! {
  #![proptest_config(cases())]

  /// What a record is known by in the store can be had without a model: the fingerprints a prepared record
  /// reports are the ones stored when it is embedded, whether its vectors are all new or partly taken from a
  /// baseline. An audit compares an export with the store on the strength of this.
  #[test]
  fn the_fingerprints_of_a_prepared_record_are_the_ones_finish_stores(
    texts in record_texts(),
    stored in record_texts(),
  ) {
    let schema = Schema::new(
      (0..3).map(|index| FieldSpec { name: format!("f{index}"), weight: 1.0, required: false, reject_below: None }).collect(),
    ).unwrap();
    let fields = |texts: &[Option<String>]| -> Vec<(String, String)> {
      texts.iter().enumerate().filter_map(|(index, text)| Some((format!("f{index}"), text.clone()?))).collect()
    };
    let embed = |prepared: &PreparedRecord, baseline: Option<&RecordVectors>| {
      let plan = prepared.plan(&schema, DIM, baseline).unwrap();
      let embedded: Vec<Vec<f32>> = plan.texts().iter().map(|text| fake_embedding(text, DIM)).collect();
      plan.finish(&schema, DIM, &embedded).unwrap()
    };
    let prepared = PreparedRecord::new(&schema, fields(&texts)).unwrap();
    let before = PreparedRecord::new(&schema, fields(&stored)).unwrap();

    let expected: BTreeMap<String, String> = texts
      .iter()
      .enumerate()
      .filter_map(|(index, text)| Some((format!("f{index}"), text_hash(&normalize_text(text.as_deref()?)?))))
      .collect();
    prop_assert_eq!(&prepared.hashes(&schema), &expected);
    prop_assert_eq!(&embed(&prepared, None).hashes, &expected);
    let baseline = embed(&before, None);
    prop_assert_eq!(&embed(&prepared, Some(&baseline)).hashes, &expected);
  }
}
