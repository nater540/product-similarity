//! The `Quantities` service: unit conversion and the comparison of quantities, for a caller with no record, and
//! the reading of an amount a caller holds as text.
//!
//! As thin as [`crate::grpc`], and thinner, since there is no engine behind it. Every answer is computed by
//! `similarity_core::quantity` from the request alone (docs/decisions.md, D47), and for `Parse` by
//! `similarity_core::amount`. That is the code a record's quantities are judged with, so what this service says
//! about two amounts is what `Evaluate` would report for them, and the two cannot come apart. Nothing here waits
//! for anything: there is no store, no model, and no lock, which is also why there is no deadline to honor and
//! nothing for readiness to check.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use similarity_core::Quantity;
use similarity_core::amount::{self, NotAnAmount};
use similarity_core::quantity::{self, Dimension, Unconvertible};
use similarity_proto::v1 as pb;
use similarity_proto::v1::quantities_server::Quantities;
use tonic::{Request, Response, Status};
use tracing::field::Empty;

use crate::grpc::{milliseconds, quantity_score};

/// The tonic service. It holds nothing: every call is answered from its request.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuantitiesService;

/// Exhaustive on purpose: a new [`Dimension`] does not compile until it has a value on the wire.
fn dimension(dimension: &Dimension) -> pb::Dimension {
  match dimension {
    Dimension::Mass => pb::Dimension::Mass,
    Dimension::Volume => pb::Dimension::Volume,
    Dimension::Fraction => pb::Dimension::Fraction,
    Dimension::Other(_) => pb::Dimension::Other,
  }
}

/// The most texts one `Parse` call reads. Not a measurement: reading a text takes well under a microsecond, so
/// this bounds the size of an answer and nothing else. A caller with more sends them in several calls.
const MAX_TEXTS: usize = 1000;

/// Exhaustive on purpose: a new reason does not compile until it has a value on the wire. A test holds each name
/// to [`NotAnAmount::code`].
fn problem(problem: NotAnAmount) -> pb::AmountProblem {
  match problem {
    NotAnAmount::Empty => pb::AmountProblem::Empty,
    NotAnAmount::NoNumber => pb::AmountProblem::NoNumber,
    NotAnAmount::NoUnit => pb::AmountProblem::NoUnit,
    NotAnAmount::AmbiguousSeparator => pb::AmountProblem::AmbiguousSeparator,
    NotAnAmount::BadNumber => pb::AmountProblem::BadNumber,
    NotAnAmount::Negative => pb::AmountProblem::Negative,
    NotAnAmount::BackwardsRange => pb::AmountProblem::BackwardsRange,
    NotAnAmount::MixedUnits => pb::AmountProblem::MixedUnits,
    NotAnAmount::Multiple => pb::AmountProblem::Multiple,
    NotAnAmount::Qualified => pb::AmountProblem::Qualified,
    NotAnAmount::UnreadableUnit => pb::AmountProblem::UnreadableUnit,
    NotAnAmount::TooLarge => pb::AmountProblem::TooLarge,
    NotAnAmount::AmbiguousName => pb::AmountProblem::AmbiguousName,
  }
}

/// A quantity as the wire holds one.
fn wire(quantity: Quantity) -> pb::Quantity {
  pb::Quantity {
    min: quantity.min,
    max: quantity.max,
    unit: quantity.unit,
  }
}

/// `Parse`: [`amount::parse`] for every text, in order. A text that is not an amount is an answer.
fn parsed(request: &pb::ParseAmountsRequest) -> Result<pb::ParseAmountsResponse, Status> {
  if request.texts.len() > MAX_TEXTS {
    return Err(Status::invalid_argument(format!(
      "{} texts in one call, and at most {MAX_TEXTS} are read: send them in several",
      request.texts.len()
    )));
  }
  let amounts = request
    .texts
    .iter()
    .map(|text| match amount::parse(text) {
      Ok(quantity) => pb::ParsedAmount {
        dimension: dimension(&quantity.dimension()).into(),
        // What `Convert` answers with no unit asked for. An amount that was read is one `check` accepts, which is
        // one that is a number in its base unit.
        normalized: quantity.in_base_unit().ok().map(wire),
        quantity: Some(wire(quantity)),
        problem: pb::AmountProblem::Unspecified.into(),
      },
      Err(refused) => pb::ParsedAmount {
        quantity: None,
        problem: problem(refused).into(),
        dimension: pb::Dimension::Unspecified.into(),
        normalized: None,
      },
    })
    .collect();
  Ok(pb::ParseAmountsResponse { amounts })
}

/// A quantity off the wire, refused unless it is a usable measurement, in the form a record keeps one in.
fn checked(what: &str, quantity: pb::Quantity) -> Result<Quantity, Status> {
  let quantity = Quantity::range(quantity.min, quantity.max, quantity.unit);
  quantity
    .check()
    .map_err(|reason| Status::invalid_argument(format!("{what}: {reason}")))?;
  Ok(quantity.trimmed())
}

/// One side of a comparison, every quantity of it checked. `side` is the request's name for it.
fn side(side: &str, quantities: HashMap<String, pb::Quantity>) -> Result<BTreeMap<String, Quantity>, Status> {
  quantities
    .into_iter()
    .map(|(name, quantity)| {
      let quantity = checked(&format!("`{side}` quantity `{name}`"), quantity)?;
      Ok((name, quantity))
    })
    .collect()
}

/// `Compare`: [`quantity::compare`] for every name either version has, in order of name.
fn compared(request: pb::CompareQuantitiesRequest) -> Result<pb::CompareQuantitiesResponse, Status> {
  let before = side("before", request.before)?;
  let after = side("after", request.after)?;
  let original = side("original", request.original)?;
  let names: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
  let quantities = names
    .into_iter()
    .filter_map(|name| quantity::compare(name, before.get(name), original.get(name), after.get(name)))
    .map(quantity_score)
    .collect();
  Ok(pb::CompareQuantitiesResponse { quantities })
}

/// `Convert`: [`Quantity::in_unit`], or [`Quantity::in_base_unit`] when no unit is asked for.
fn converted(request: pb::ConvertQuantityRequest) -> Result<pb::ConvertQuantityResponse, Status> {
  let Some(quantity) = request.quantity else {
    return Err(Status::invalid_argument("`quantity` is required"));
  };
  let quantity = checked("`quantity`", quantity)?;
  let to_unit = request.to_unit.as_deref();
  if to_unit.is_some_and(|unit| unit.trim().is_empty()) {
    return Err(Status::invalid_argument(
      "`to_unit` is blank; leave it out for the base unit",
    ));
  }
  let from = quantity.dimension();
  let (to, outcome) = match to_unit {
    Some(unit) => (quantity::unit(unit).0, quantity.in_unit(unit)),
    None => (from.clone(), quantity.in_base_unit()),
  };
  let converted = match outcome {
    Ok(converted) => Some(wire(converted)),
    // An answer, like `QUANTITY_CHANGE_INCOMPARABLE` is: the two dimensions in the response say why.
    Err(Unconvertible::OtherDimension { .. }) => None,
    Err(error @ Unconvertible::TooLarge { .. }) => return Err(Status::invalid_argument(error.to_string())),
  };
  Ok(pb::ConvertQuantityResponse {
    dimension: dimension(&from).into(),
    to_dimension: dimension(&to).into(),
    converted,
  })
}

/// `ListUnits`: the table, as [`quantity::units`] lists it.
fn listed() -> pb::ListUnitsResponse {
  pb::ListUnitsResponse {
    units: quantity::units()
      .map(|entry| pb::UnitDescription {
        spellings: entry.spellings.iter().map(|spelling| (*spelling).to_owned()).collect(),
        dimension: dimension(&entry.dimension).into(),
        base_unit: entry.dimension.base_unit().to_owned(),
        factor: entry.factor,
      })
      .collect(),
  }
}

#[tonic::async_trait]
impl Quantities for QuantitiesService {
  async fn compare(
    &self,
    request: Request<pb::CompareQuantitiesRequest>,
  ) -> Result<Response<pb::CompareQuantitiesResponse>, Status> {
    let started = Instant::now();
    // How many were compared, and never a name, a unit, or an amount (D40).
    let span = tracing::debug_span!("compare_quantities", quantities = Empty);
    let response = span.in_scope(|| compared(request.into_inner()))?;
    let quantities = i64::try_from(response.quantities.len()).unwrap_or(i64::MAX);
    span.record("quantities", quantities);
    tracing::info!(quantities, elapsed_ms = milliseconds(started), "compare_quantities");
    Ok(Response::new(response))
  }

  async fn convert(
    &self,
    request: Request<pb::ConvertQuantityRequest>,
  ) -> Result<Response<pb::ConvertQuantityResponse>, Status> {
    let started = Instant::now();
    let span = tracing::debug_span!("convert_quantity", converted = Empty);
    let response = span.in_scope(|| converted(request.into_inner()))?;
    let converted = response.converted.is_some();
    span.record("converted", converted);
    tracing::info!(converted, elapsed_ms = milliseconds(started), "convert_quantity");
    Ok(Response::new(response))
  }

  async fn list_units(&self, _: Request<pb::ListUnitsRequest>) -> Result<Response<pb::ListUnitsResponse>, Status> {
    Ok(Response::new(listed()))
  }

  async fn parse(
    &self,
    request: Request<pb::ParseAmountsRequest>,
  ) -> Result<Response<pb::ParseAmountsResponse>, Status> {
    let started = Instant::now();
    // How many texts there were and how many of them were amounts, and never a text (D40).
    let span = tracing::debug_span!("parse_amounts", texts = Empty, amounts = Empty);
    let response = span.in_scope(|| parsed(request.get_ref()))?;
    let count = |number: usize| i64::try_from(number).unwrap_or(i64::MAX);
    let texts = count(response.amounts.len());
    let amounts = count(response.amounts.iter().filter(|each| each.quantity.is_some()).count());
    span.record("texts", texts);
    span.record("amounts", amounts);
    tracing::info!(texts, amounts, elapsed_ms = milliseconds(started), "parse_amounts");
    Ok(Response::new(response))
  }
}

#[cfg(test)]
mod tests {
  use tonic::Code;

  use super::*;

  fn amount(value: f64, unit: &str) -> pb::Quantity {
    pb::Quantity {
      min: value,
      max: value,
      unit: unit.to_owned(),
    }
  }

  fn named(quantities: &[(&str, pb::Quantity)]) -> HashMap<String, pb::Quantity> {
    quantities
      .iter()
      .map(|(name, quantity)| ((*name).to_owned(), quantity.clone()))
      .collect()
  }

  fn compare(
    before: &[(&str, pb::Quantity)],
    after: &[(&str, pb::Quantity)],
    original: &[(&str, pb::Quantity)],
  ) -> Result<Vec<pb::QuantityScore>, Status> {
    compared(pb::CompareQuantitiesRequest {
      before: named(before),
      after: named(after),
      original: named(original),
    })
    .map(|response| response.quantities)
  }

  fn convert(quantity: pb::Quantity, to_unit: Option<&str>) -> Result<pb::ConvertQuantityResponse, Status> {
    converted(pb::ConvertQuantityRequest {
      quantity: Some(quantity),
      to_unit: to_unit.map(str::to_owned),
    })
  }

  #[test]
  fn the_same_amount_in_another_unit_is_unchanged_and_an_eighth_to_an_ounce_is_a_factor_of_eight() {
    let scores = compare(
      &[("thc", amount(100.0, "mg")), ("weight", amount(3.5, "g"))],
      &[("thc", amount(0.1, "g")), ("weight", amount(28.0, "g"))],
      &[],
    )
    .unwrap();
    // In order of name, whatever order the maps came in.
    assert_eq!(
      scores.iter().map(|score| score.field.as_str()).collect::<Vec<_>>(),
      ["thc", "weight"]
    );
    let (thc, weight) = (&scores[0], &scores[1]);
    assert_eq!(thc.change(), pb::QuantityChange::Unchanged);
    assert_eq!(thc.factor, Some(1.0));
    assert_eq!(
      (thc.previous_unit.as_deref(), thc.unit.as_deref()),
      (Some("mg"), Some("g"))
    );
    assert_eq!(weight.change(), pb::QuantityChange::Modified);
    assert_eq!(weight.factor, Some(8.0));
    assert!(
      scores
        .iter()
        .all(|score| !score.beyond_limit && !score.incomparable && !score.zero_crossing)
    );
  }

  #[test]
  fn a_name_on_one_side_only_is_added_or_removed_and_a_name_on_neither_is_not_answered() {
    let scores = compare(
      &[("volume", amount(30.0, "ml"))],
      &[("weight", amount(1.0, "g"))],
      // An original for a name neither version has says nothing about either version.
      &[("cbd", amount(5.0, "mg"))],
    )
    .unwrap();
    assert_eq!(scores.len(), 2);
    assert_eq!(
      (scores[0].field.as_str(), scores[0].change()),
      ("volume", pb::QuantityChange::Removed)
    );
    assert_eq!(
      (scores[0].previous_unit.as_deref(), scores[0].unit.as_deref()),
      (Some("ml"), None)
    );
    assert_eq!(
      (scores[1].field.as_str(), scores[1].change()),
      ("weight", pb::QuantityChange::Added)
    );
    assert_eq!(
      (scores[1].previous_unit.as_deref(), scores[1].unit.as_deref()),
      (None, Some("g"))
    );
    assert!(scores.iter().all(|score| score.factor.is_none()));

    assert!(compare(&[], &[], &[("cbd", amount(5.0, "mg"))]).unwrap().is_empty());
  }

  /// What `Evaluate` does for a record whose quantity was blanked and filled in again (D8, D16).
  #[test]
  fn a_quantity_that_comes_back_is_compared_with_the_original() {
    let scores = compare(&[], &[("weight", amount(28.0, "g"))], &[("weight", amount(3.5, "g"))]).unwrap();
    assert_eq!(scores[0].change(), pb::QuantityChange::Restored);
    assert_eq!(scores[0].factor, Some(8.0));
    // With a value before it, the original is not what it is compared with.
    let scores = compare(
      &[("weight", amount(28.0, "g"))],
      &[("weight", amount(28.0, "g"))],
      &[("weight", amount(3.5, "g"))],
    )
    .unwrap();
    assert_eq!(scores[0].change(), pb::QuantityChange::Unchanged);
  }

  #[test]
  fn units_that_measure_different_things_and_a_move_from_zero_are_answers_and_not_errors() {
    let scores = compare(
      &[
        ("cbd", amount(0.0, "%")),
        ("count", amount(2.0, "pack")),
        ("thc", amount(0.1, "g")),
      ],
      &[
        ("cbd", amount(0.1, "%")),
        ("count", amount(2.0, " PACK ")),
        ("thc", amount(20.0, "%")),
      ],
      &[],
    )
    .unwrap();
    let (cbd, count, thc) = (&scores[0], &scores[1], &scores[2]);
    assert!(cbd.zero_crossing && cbd.factor.is_none());
    assert_eq!(cbd.change(), pb::QuantityChange::Modified);
    // A unit that is not converted is comparable with itself.
    assert_eq!(count.change(), pb::QuantityChange::Unchanged);
    assert_eq!(thc.change(), pb::QuantityChange::Incomparable);
    assert!(thc.incomparable && thc.factor.is_none());
  }

  #[test]
  fn an_amount_that_is_not_a_measurement_is_refused_by_side_and_name() {
    for (broken, reason) in [
      (amount(-1.0, "g"), "cannot be negative"),
      (amount(f64::NAN, "g"), "not a finite number"),
      (amount(f64::INFINITY, "g"), "not a finite number"),
      (amount(1.0, "  "), "`unit` is blank"),
      (amount(1e305, "kg"), "too large to compare"),
      (
        pb::Quantity {
          min: 2.0,
          max: 1.0,
          unit: "g".to_owned(),
        },
        "is above `max`",
      ),
    ] {
      let good = amount(1.0, "g");
      for (side, error) in [
        (
          "before",
          compare(&[("weight", broken.clone())], &[("weight", good.clone())], &[]),
        ),
        (
          "after",
          compare(&[("weight", good.clone())], &[("weight", broken.clone())], &[]),
        ),
        (
          "original",
          compare(&[], &[("weight", good.clone())], &[("weight", broken.clone())]),
        ),
      ] {
        let error = error.unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
        assert!(
          error.message().starts_with(&format!("`{side}` quantity `weight`: ")),
          "{}",
          error.message()
        );
        assert!(error.message().contains(reason), "{}", error.message());
      }
    }
  }

  #[test]
  fn a_quantity_is_converted_to_the_unit_asked_for_or_to_its_base_unit() {
    let answer = convert(amount(3.5, " G "), Some(" mg ")).unwrap();
    assert_eq!(answer.converted, Some(amount(3500.0, "mg")));
    assert_eq!(
      (answer.dimension(), answer.to_dimension()),
      (pb::Dimension::Mass, pb::Dimension::Mass)
    );

    let ranged = pb::Quantity {
      min: 0.5,
      max: 2.0,
      unit: "l".to_owned(),
    };
    let answer = convert(ranged, None).unwrap();
    assert_eq!(
      answer.converted,
      Some(pb::Quantity {
        min: 500.0,
        max: 2000.0,
        unit: "ml".to_owned(),
      })
    );
    assert_eq!(
      (answer.dimension(), answer.to_dimension()),
      (pb::Dimension::Volume, pb::Dimension::Volume)
    );

    let answer = convert(amount(20.0, "percent"), None).unwrap();
    assert_eq!(answer.converted, Some(amount(20.0, "%")));
    assert_eq!(answer.dimension(), pb::Dimension::Fraction);

    // A unit that is not converted is its own base unit, normalized, and is itself under any case and spacing.
    let answer = convert(amount(2.0, " Pre  Rolls "), None).unwrap();
    assert_eq!(answer.converted, Some(amount(2.0, "pre rolls")));
    assert_eq!(
      (answer.dimension(), answer.to_dimension()),
      (pb::Dimension::Other, pb::Dimension::Other)
    );
    assert_eq!(
      convert(amount(2.0, "Pack"), Some("pack")).unwrap().converted,
      Some(amount(2.0, "pack"))
    );
  }

  #[test]
  fn units_that_do_not_convert_are_answered_with_their_dimensions_and_no_quantity() {
    let answer = convert(amount(20.0, "%"), Some("mg")).unwrap();
    assert_eq!(answer.converted, None);
    assert_eq!(
      (answer.dimension(), answer.to_dimension()),
      (pb::Dimension::Fraction, pb::Dimension::Mass)
    );

    let answer = convert(amount(1.0, "g"), Some("pack")).unwrap();
    assert_eq!(answer.converted, None);
    assert_eq!(
      (answer.dimension(), answer.to_dimension()),
      (pb::Dimension::Mass, pb::Dimension::Other)
    );

    // Two units that are not converted are both `OTHER`, and still not one unit.
    let answer = convert(amount(1.0, "pack"), Some("each")).unwrap();
    assert_eq!(answer.converted, None);
    assert_eq!(
      (answer.dimension(), answer.to_dimension()),
      (pb::Dimension::Other, pb::Dimension::Other)
    );
  }

  #[test]
  fn a_conversion_that_cannot_be_asked_for_is_refused() {
    let missing = converted(pb::ConvertQuantityRequest {
      quantity: None,
      to_unit: None,
    })
    .unwrap_err();
    assert_eq!(
      (missing.code(), missing.message()),
      (Code::InvalidArgument, "`quantity` is required")
    );

    let blank = convert(amount(1.0, "g"), Some("  ")).unwrap_err();
    assert_eq!(blank.code(), Code::InvalidArgument);
    assert!(blank.message().starts_with("`to_unit` is blank"), "{}", blank.message());

    let negative = convert(amount(-1.0, "g"), None).unwrap_err();
    assert_eq!(negative.code(), Code::InvalidArgument);
    assert!(negative.message().starts_with("`quantity`: "), "{}", negative.message());

    // A number of milligrams, and not a number of micrograms.
    let huge = convert(amount(1e300, "kg"), Some("mcg")).unwrap_err();
    assert_eq!(huge.code(), Code::InvalidArgument);
    assert!(
      huge.message().contains("too large to write in `mcg`"),
      "{}",
      huge.message()
    );
    assert!(convert(amount(1e300, "kg"), Some("g")).is_ok());
  }

  #[test]
  fn every_listed_unit_converts_to_its_base_unit_by_its_factor() {
    let units = listed().units;
    assert_eq!(units.len(), quantity::units().count());
    for unit in &units {
      assert_ne!(unit.dimension(), pb::Dimension::Other);
      assert_ne!(unit.dimension(), pb::Dimension::Unspecified);
      for spelling in &unit.spellings {
        let answer = convert(amount(1.0, spelling), None).unwrap();
        assert_eq!(
          answer.converted,
          Some(amount(unit.factor, &unit.base_unit)),
          "{spelling}"
        );
        assert_eq!(answer.dimension(), unit.dimension());
      }
    }
    let grams = units
      .iter()
      .find(|unit| unit.spellings.iter().any(|spelling| spelling == "g"))
      .unwrap();
    assert_eq!(
      (grams.base_unit.as_str(), grams.factor, grams.dimension()),
      ("mg", 1000.0, pb::Dimension::Mass)
    );
  }

  #[test]
  fn every_dimension_has_a_wire_value_of_its_own() {
    let wire: Vec<i32> = [
      Dimension::Mass,
      Dimension::Volume,
      Dimension::Fraction,
      Dimension::Other("pack".to_owned()),
    ]
    .iter()
    .map(|each| i32::from(dimension(each)))
    .collect();
    let distinct: BTreeSet<i32> = wire.iter().copied().collect();
    assert_eq!(distinct.len(), wire.len());
    assert!(!wire.contains(&i32::from(pb::Dimension::Unspecified)));
  }

  fn parse(texts: &[&str]) -> Vec<pb::ParsedAmount> {
    parsed(&pb::ParseAmountsRequest {
      texts: texts.iter().map(|text| (*text).to_owned()).collect(),
    })
    .unwrap()
    .amounts
  }

  #[test]
  fn texts_are_read_in_order_and_one_that_is_not_an_amount_is_an_answer() {
    let amounts = parse(&["3.5g", "3,5 g", " 18 - 22 % ", "", "2 pack", "1 1/2 oz"]);
    assert_eq!(amounts.len(), 6);
    assert_eq!(amounts[0].quantity, Some(amount(3.5, "g")));
    assert_eq!(
      (amounts[0].problem(), amounts[0].dimension()),
      (pb::AmountProblem::Unspecified, pb::Dimension::Mass)
    );
    assert_eq!(amounts[1].quantity, None);
    assert_eq!(
      (amounts[1].problem(), amounts[1].dimension()),
      (pb::AmountProblem::AmbiguousSeparator, pb::Dimension::Unspecified)
    );
    assert_eq!(
      amounts[2].quantity,
      Some(pb::Quantity {
        min: 18.0,
        max: 22.0,
        unit: "%".to_owned(),
      })
    );
    assert_eq!(amounts[2].dimension(), pb::Dimension::Fraction);
    assert_eq!(
      (amounts[3].quantity.as_ref(), amounts[3].problem()),
      (None, pb::AmountProblem::Empty)
    );
    // A unit that is not converted is an amount all the same, and the answer says which kind it is.
    assert_eq!(amounts[4].quantity, Some(amount(2.0, "pack")));
    assert_eq!(
      (amounts[4].problem(), amounts[4].dimension()),
      (pb::AmountProblem::Unspecified, pb::Dimension::Other)
    );
    assert_eq!(amounts[5].quantity, Some(amount(1.5, "oz")));
    assert!(parse(&[]).is_empty());
  }

  /// A caller that wants every weight in one unit gets it with the amount: what `Convert` would answer.
  #[test]
  fn a_parsed_amount_comes_with_itself_in_the_unit_its_dimension_is_compared_in() {
    let amounts = parse(&[
      "3.5 g",
      "1/8 oz",
      "3500 mg",
      "12 Fl. Oz.",
      "18-22 percent",
      "2  Pre Rolls",
      "3,5 g",
    ]);
    let normalized = |index: usize| amounts[index].normalized.clone();
    assert_eq!(normalized(0), Some(amount(3500.0, "mg")));
    assert_eq!(normalized(1), Some(amount(3_543.690_390_625, "mg")));
    // One amount under two spellings is one normalized amount, and what was written is still there.
    assert_eq!(normalized(2), normalized(0));
    assert_eq!(amounts[2].quantity, Some(amount(3500.0, "mg")));
    assert_eq!(normalized(3), Some(amount(12.0 * 29.573_529_562_5, "ml")));
    assert_eq!(
      normalized(4),
      Some(pb::Quantity {
        min: 18.0,
        max: 22.0,
        unit: "%".to_owned(),
      })
    );
    // A unit that is not converted is its own, in the spelling it is matched in.
    assert_eq!(amounts[5].quantity, Some(amount(2.0, "Pre Rolls")));
    assert_eq!(normalized(5), Some(amount(2.0, "pre rolls")));
    assert_eq!(normalized(6), None);
    for parsed in &amounts {
      if let Some(quantity) = &parsed.quantity {
        assert_eq!(parsed.normalized, convert(quantity.clone(), None).unwrap().converted);
        // Either may be sent as a record's quantity: they are the same amount.
        let same = compare(
          &[("q", quantity.clone())],
          &[("q", parsed.normalized.clone().unwrap())],
          &[],
        )
        .unwrap();
        assert_eq!(same[0].change(), pb::QuantityChange::Unchanged);
      }
    }
  }

  /// What `Parse` reads can be sent as a record's quantity as it is, and compares as the amount it is.
  #[test]
  fn a_parsed_amount_is_a_quantity_the_other_calls_accept() {
    let amounts = parse(&["1/8 oz", "3.5 g", "100mg", "0.1 g"]);
    let quantity = |index: usize| amounts[index].quantity.clone().unwrap();
    let scores = compare(
      &[("thc", quantity(2)), ("weight", quantity(1))],
      &[("thc", quantity(3)), ("weight", quantity(0))],
      &[],
    )
    .unwrap();
    assert_eq!(scores[0].change(), pb::QuantityChange::Unchanged);
    // An eighth of an ounce is 3.54 g, not 3.5 g: a small change, and reported as one.
    assert_eq!(scores[1].change(), pb::QuantityChange::Modified);
    assert!(
      (scores[1].factor.unwrap() - 1.0125).abs() < 1e-4,
      "{:?}",
      scores[1].factor
    );
  }

  #[test]
  fn more_texts_than_one_call_reads_are_refused_and_exactly_that_many_are_read() {
    let texts = |count: usize| pb::ParseAmountsRequest {
      texts: vec!["1 g".to_owned(); count],
    };
    assert_eq!(parsed(&texts(MAX_TEXTS)).unwrap().amounts.len(), 1000);
    let refused = parsed(&texts(MAX_TEXTS + 1)).unwrap_err();
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert!(
      refused.message().starts_with("1001 texts in one call"),
      "{}",
      refused.message()
    );
  }

  /// The codes the offline tools count are an interface, and the API's enum is the same names, so a refusal
  /// read from a response and one read from a survey are joined without a table.
  #[test]
  fn every_reason_a_text_is_refused_for_has_a_wire_value_named_after_its_code() {
    let wire: BTreeSet<i32> = NotAnAmount::ALL.iter().map(|each| i32::from(problem(*each))).collect();
    assert_eq!(wire.len(), NotAnAmount::ALL.len());
    assert!(!wire.contains(&i32::from(pb::AmountProblem::Unspecified)));
    for refused in NotAnAmount::ALL {
      assert_eq!(
        problem(refused).as_str_name(),
        format!("AMOUNT_PROBLEM_{}", refused.code().to_uppercase())
      );
    }
  }

  /// One line of `data/amounts.cases.jsonl`, which `similarity_core::amount` is held to as well: the service
  /// and the function cannot differ, and a client in another language can hold a server to the same file.
  #[derive(Debug, serde::Deserialize)]
  #[serde(deny_unknown_fields)]
  struct Case {
    text: String,
    min: Option<f64>,
    max: Option<f64>,
    unit: Option<String>,
    refused: Option<String>,
  }

  #[test]
  fn the_service_reads_every_case_in_the_file_as_the_file_says() {
    let cases: Vec<Case> = include_str!("../../../data/amounts.cases.jsonl")
      .lines()
      .filter(|line| !(line.trim().is_empty() || line.starts_with('#')))
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();
    assert!(cases.len() > 150);
    // In calls of no more than one call reads, as a client would send them.
    for batch in cases.chunks(MAX_TEXTS) {
      let texts: Vec<&str> = batch.iter().map(|case| case.text.as_str()).collect();
      for (case, answer) in batch.iter().zip(parse(&texts)) {
        match (&case.refused, &answer.quantity) {
          (None, Some(quantity)) => {
            assert_eq!(
              Some(quantity.min.to_bits()),
              case.min.map(f64::to_bits),
              "{:?}",
              case.text
            );
            assert_eq!(
              Some(quantity.max.to_bits()),
              case.max.map(f64::to_bits),
              "{:?}",
              case.text
            );
            assert_eq!(Some(&quantity.unit), case.unit.as_ref(), "{:?}", case.text);
            assert_eq!(answer.problem(), pb::AmountProblem::Unspecified, "{:?}", case.text);
            assert_ne!(answer.dimension(), pb::Dimension::Unspecified, "{:?}", case.text);
            assert_eq!(
              answer.normalized,
              convert(quantity.clone(), None).unwrap().converted,
              "{:?}",
              case.text
            );
          }
          (Some(code), None) => {
            assert_eq!(
              answer.problem().as_str_name(),
              format!("AMOUNT_PROBLEM_{}", code.to_uppercase()),
              "{:?}",
              case.text
            );
            assert_eq!(answer.dimension(), pb::Dimension::Unspecified, "{:?}", case.text);
            assert_eq!(answer.normalized, None, "{:?}", case.text);
          }
          other => panic!("{:?}: {other:?}", case.text),
        }
      }
    }
  }

  #[tokio::test]
  async fn the_service_answers_each_call_from_its_request_alone() {
    let service = QuantitiesService;
    let compared = service
      .compare(Request::new(pb::CompareQuantitiesRequest {
        before: named(&[("weight", amount(1.0, "oz"))]),
        after: named(&[("weight", amount(28.349_523_125, "g"))]),
        original: HashMap::new(),
      }))
      .await
      .unwrap()
      .into_inner();
    assert_eq!(compared.quantities[0].change(), pb::QuantityChange::Unchanged);

    let converted = service
      .convert(Request::new(pb::ConvertQuantityRequest {
        quantity: Some(amount(1.0, "lb")),
        to_unit: Some("oz".to_owned()),
      }))
      .await
      .unwrap()
      .into_inner();
    let ounces = converted.converted.unwrap();
    assert!((ounces.min - 16.0).abs() < 1e-9, "{}", ounces.min);

    let units = service
      .list_units(Request::new(pb::ListUnitsRequest {}))
      .await
      .unwrap()
      .into_inner();
    assert!(!units.units.is_empty());

    let read = service
      .parse(Request::new(pb::ParseAmountsRequest {
        texts: vec![
          "½ oz".to_owned(),
          "about half an ounce".to_owned(),
          "an eighth".to_owned(),
        ],
      }))
      .await
      .unwrap()
      .into_inner();
    assert_eq!(read.amounts[0].quantity, Some(amount(0.5, "oz")));
    assert_eq!(read.amounts[1].problem(), pb::AmountProblem::NoNumber);
    assert_eq!(read.amounts[2].quantity, Some(amount(3.5, "g")));
    assert_eq!(read.amounts[2].normalized, Some(amount(3500.0, "mg")));
    let too_many = service
      .parse(Request::new(pb::ParseAmountsRequest {
        texts: vec![String::new(); 1001],
      }))
      .await
      .unwrap_err();
    assert_eq!(too_many.code(), Code::InvalidArgument);

    let refused = service
      .convert(Request::new(pb::ConvertQuantityRequest::default()))
      .await
      .unwrap_err();
    assert_eq!(refused.code(), Code::InvalidArgument);
  }
}
