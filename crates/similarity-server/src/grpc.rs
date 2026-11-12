//! The gRPC surface: protobuf in, [`Engine`] call, protobuf out.
//!
//! Nothing in this module decides anything. It translates messages, maps [`EngineError`] onto status codes a client
//! can act on, and logs one line per request. Keeping it that thin is what lets the engine's tests stand in for
//! tests of the service.

use std::sync::Arc;
use std::time::Instant;

use similarity_core::matching::Decision;
use similarity_core::{FieldChange, FieldScore, Quantity, QuantityChange, QuantityScore, Reason};
use similarity_proto::v1 as pb;
use similarity_proto::v1::record_similarity_server::RecordSimilarity;
use similarity_store::VectorStore;
use tonic::{Request, Response, Status};

use crate::engine::{
  Engine, EngineError, EvaluateInput, FindSimilarInput, MatchInput, RecordInput, SimilarTo, Verdict, WriteMode,
};
use crate::worker::TextEmbedder;

/// The tonic service wrapping an [`Engine`].
#[derive(Debug)]
pub struct GrpcService<S, E> {
  engine: Arc<Engine<S, E>>,
}

impl<S, E> GrpcService<S, E> {
  /// Wraps an engine.
  pub fn new(engine: Arc<Engine<S, E>>) -> Self {
    Self { engine }
  }
}

impl From<EngineError> for Status {
  fn from(error: EngineError) -> Self {
    match error {
      EngineError::InvalidArgument(message) => Self::invalid_argument(message),
      EngineError::NotFound(message) => Self::not_found(message),
      EngineError::FailedPrecondition(message) => Self::failed_precondition(message),
      EngineError::Overloaded(message) => Self::resource_exhausted(message),
      EngineError::Unavailable(message) => {
        tracing::warn!(%message, "dependency unavailable");
        Self::unavailable(message)
      }
      EngineError::Internal(message) => {
        tracing::error!(%message, "internal error");
        Self::internal(message)
      }
    }
  }
}

fn record_input(record: pb::Record) -> RecordInput {
  RecordInput {
    id: record.id,
    tenant: record.tenant,
    fields: record.fields.into_iter().collect(),
    // Passed on exactly as sent. Whether a quantity is a usable measurement is the engine's to decide.
    quantities: record
      .quantities
      .into_iter()
      .map(|(name, quantity)| (name, Quantity::range(quantity.min, quantity.max, quantity.unit)))
      .collect(),
  }
}

/// Decodes the wire value of `EvaluateRequest.mode`.
///
/// A value this build does not know is refused rather than defaulted. The default writes, and a newer client
/// asking for a mode that does not exist yet (a second kind of preview, say) must not get a write instead.
fn write_mode(mode: i32) -> Result<WriteMode, Status> {
  match pb::WriteMode::try_from(mode) {
    Ok(pb::WriteMode::Unspecified | pb::WriteMode::Enforce) => Ok(WriteMode::Enforce),
    Ok(pb::WriteMode::DryRun) => Ok(WriteMode::DryRun),
    Ok(pb::WriteMode::Force) => Ok(WriteMode::Force),
    Err(_) => Err(Status::invalid_argument(format!(
      "`mode` is {mode}, which is not a write mode this server knows"
    ))),
  }
}

fn verdict(verdict: Verdict) -> pb::Verdict {
  match verdict {
    Verdict::Embedded => pb::Verdict::Embedded,
    Verdict::Created => pb::Verdict::Created,
    Verdict::Accepted => pb::Verdict::Accepted,
    Verdict::Rejected => pb::Verdict::Rejected,
  }
}

fn field_change(change: FieldChange) -> pb::FieldChange {
  match change {
    FieldChange::Unchanged => pb::FieldChange::Unchanged,
    FieldChange::Modified => pb::FieldChange::Modified,
    FieldChange::Added => pb::FieldChange::Added,
    FieldChange::Removed => pb::FieldChange::Removed,
    FieldChange::Restored => pb::FieldChange::Restored,
  }
}

fn field_score(score: FieldScore) -> pb::FieldScore {
  pb::FieldScore {
    field: score.field,
    change: field_change(score.change).into(),
    similarity: score.similarity,
    weight: score.weight,
    below_floor: score.below_floor,
  }
}

fn match_decision(decision: Decision) -> pb::MatchDecision {
  match decision {
    Decision::NotJudged => pb::MatchDecision::NotJudged,
    Decision::Matched => pb::MatchDecision::Matched,
    Decision::NoMatch => pb::MatchDecision::NoMatch,
  }
}

fn quantity_change(change: QuantityChange) -> pb::QuantityChange {
  match change {
    QuantityChange::Unchanged => pb::QuantityChange::Unchanged,
    QuantityChange::Modified => pb::QuantityChange::Modified,
    QuantityChange::Incomparable => pb::QuantityChange::Incomparable,
    QuantityChange::Added => pb::QuantityChange::Added,
    QuantityChange::Removed => pb::QuantityChange::Removed,
    QuantityChange::Restored => pb::QuantityChange::Restored,
  }
}

fn quantity_score(score: QuantityScore) -> pb::QuantityScore {
  pb::QuantityScore {
    field: score.field,
    change: quantity_change(score.change).into(),
    factor: score.factor,
    zero_crossing: score.zero_crossing,
    incomparable: score.incomparable,
    previous_unit: score.previous_unit,
    unit: score.unit,
    beyond_limit: score.beyond_limit,
  }
}

/// Exhaustive on purpose: a new kind of [`Reason`] does not compile until it has a value on the wire.
fn reason_code(reason: &Reason) -> pb::ReasonCode {
  match reason {
    Reason::BelowThreshold { .. } => pb::ReasonCode::BelowThreshold,
    Reason::FieldBelowFloor { .. } => pb::ReasonCode::FieldBelowFloor,
    Reason::AnchorDrift { .. } => pb::ReasonCode::AnchorDrift,
    Reason::QuantityBeyondFactor { .. } => pb::ReasonCode::QuantityBeyondFactor,
    Reason::QuantityIncomparable { .. } => pb::ReasonCode::QuantityIncomparable,
    Reason::MeasureChanged => pb::ReasonCode::MeasureChanged,
  }
}

fn reason_detail(reason: &Reason) -> pb::ReasonDetail {
  pb::ReasonDetail {
    code: reason_code(reason).into(),
    subject: reason.subject().map(str::to_owned),
    value: reason.value(),
    limit: reason.limit(),
  }
}

/// A similarity for a log line: widened and rounded to six places, so that a 32-bit value does not print its
/// conversion noise (`0.5670740008354187`). The response carries the value itself; this is only what is logged.
fn six(value: f32) -> f64 {
  (f64::from(value) * 1e6).round() / 1e6
}

/// How long a request has taken so far, in milliseconds to two places: from the first line of its handler to its
/// log line, which is everything but the framing of the reply.
fn milliseconds(started: Instant) -> f64 {
  (started.elapsed().as_secs_f64() * 100_000.0).round() / 100.0
}

fn width(value: usize) -> u32 {
  u32::try_from(value).unwrap_or(u32::MAX)
}

#[tonic::async_trait]
impl<S: VectorStore, E: TextEmbedder> RecordSimilarity for GrpcService<S, E> {
  async fn evaluate(&self, request: Request<pb::EvaluateRequest>) -> Result<Response<pb::EvaluateResponse>, Status> {
    let started = Instant::now();
    let request = request.into_inner();
    let mode = write_mode(request.mode)?;
    let Some(record) = request.record else {
      return Err(Status::invalid_argument("`record` is required"));
    };
    let id = record.id.clone();

    let output = self
      .engine
      .evaluate(EvaluateInput {
        record: record_input(record),
        mode,
        threshold: request.threshold,
      })
      .await?;
    tracing::info!(
      id = id.as_deref().unwrap_or(""),
      verdict = ?output.verdict,
      similarity = output.similarity.map(six),
      anchor_similarity = output.anchor_similarity.map(six),
      stored = output.stored,
      revision = output.revision,
      measure_changed = output.measure_changed,
      elapsed_ms = milliseconds(started),
      "evaluate"
    );

    Ok(Response::new(pb::EvaluateResponse {
      verdict: verdict(output.verdict).into(),
      similarity: output.similarity,
      threshold: output.threshold,
      fields: output.fields.into_iter().map(field_score).collect(),
      anchor_similarity: output.anchor_similarity,
      stored: output.stored,
      // The sentence and the data are two renderings of one list, so they cannot fall out of step.
      reasons: output.reasons.iter().map(ToString::to_string).collect(),
      reason_details: output.reasons.iter().map(reason_detail).collect(),
      vector: if request.return_vector {
        output.vector
      } else {
        Vec::new()
      },
      revision: output.revision,
      quantities: output.quantities.into_iter().map(quantity_score).collect(),
      measure_changed: output.measure_changed,
    }))
  }

  async fn find_similar(
    &self,
    request: Request<pb::FindSimilarRequest>,
  ) -> Result<Response<pb::FindSimilarResponse>, Status> {
    let started = Instant::now();
    let request = request.into_inner();
    let query = match request.query {
      Some(pb::find_similar_request::Query::Id(id)) => SimilarTo::Id(id),
      Some(pb::find_similar_request::Query::Record(record)) => SimilarTo::Record(record_input(record)),
      None => return Err(Status::invalid_argument("one of `id` or `record` is required")),
    };
    let neighbors = self
      .engine
      .find_similar(FindSimilarInput {
        query,
        limit: usize::try_from(request.limit).unwrap_or(usize::MAX),
        tenant: request.tenant,
        min_similarity: request.min_similarity,
      })
      .await?;
    tracing::info!(
      neighbors = neighbors.len(),
      elapsed_ms = milliseconds(started),
      "find_similar"
    );

    Ok(Response::new(pb::FindSimilarResponse {
      neighbors: neighbors
        .into_iter()
        .map(|neighbor| pb::Neighbor {
          id: neighbor.id,
          tenant: neighbor.tenant,
          similarity: neighbor.similarity,
        })
        .collect(),
    }))
  }

  async fn delete(&self, request: Request<pb::DeleteRequest>) -> Result<Response<pb::DeleteResponse>, Status> {
    let started = Instant::now();
    let pb::DeleteRequest { id, tenant } = request.into_inner();
    let deleted = self.engine.delete(id.clone(), tenant).await?;
    tracing::info!(%id, deleted, elapsed_ms = milliseconds(started), "delete");
    Ok(Response::new(pb::DeleteResponse { deleted }))
  }

  async fn r#match(&self, request: Request<pb::MatchRequest>) -> Result<Response<pb::MatchResponse>, Status> {
    let started = Instant::now();
    let request = request.into_inner();
    let Some(listing) = request.listing else {
      return Err(Status::invalid_argument("`listing` is required"));
    };
    let output = self
      .engine
      .match_listing(MatchInput {
        listing: record_input(listing),
        limit: usize::try_from(request.limit).unwrap_or(usize::MAX),
        tenant: request.tenant,
        threshold: request.threshold,
      })
      .await?;
    tracing::info!(
      candidates = output.candidates.len(),
      fetched = output.fetched,
      top_similarity = output
        .candidates
        .first()
        .map(|candidate| six(candidate.comparison.similarity)),
      decision = ?output.decision,
      elapsed_ms = milliseconds(started),
      "match"
    );

    Ok(Response::new(pb::MatchResponse {
      candidates: output
        .candidates
        .into_iter()
        .map(|candidate| pb::MatchCandidate {
          id: candidate.id,
          tenant: candidate.tenant,
          similarity: candidate.comparison.similarity,
          search_similarity: candidate.search_similarity,
          fields: candidate.comparison.fields.into_iter().map(field_score).collect(),
          quantities: candidate.quantities.into_iter().map(quantity_score).collect(),
        })
        .collect(),
      decision: match_decision(output.decision).into(),
      threshold: output.threshold,
    }))
  }

  async fn describe(&self, _: Request<pb::DescribeRequest>) -> Result<Response<pb::DescribeResponse>, Status> {
    let description = self.engine.describe();
    Ok(Response::new(pb::DescribeResponse {
      model_fingerprint: description.model_fingerprint,
      dimension: width(description.dimension),
      composite_dimension: width(description.composite_dimension),
      fields: description
        .schema
        .fields()
        .iter()
        .map(|field| pb::FieldDescription {
          name: field.name.clone(),
          weight: field.weight,
          required: field.required,
          reject_below: field.reject_below,
        })
        .collect(),
      threshold: description.policy.threshold,
      anchor_threshold: description.policy.anchor_threshold,
      quantities: description
        .schema
        .quantities()
        .iter()
        .map(|quantity| pb::QuantityDescription {
          name: quantity.name.clone(),
          reject_above_factor: quantity.reject_above_factor,
          reject_incomparable: quantity.reject_incomparable,
        })
        .collect(),
      reject_on_measure_change: description.policy.reject_on_measure_change,
      match_threshold: description.matching.threshold,
      match_overfetch: width(description.matching.overfetch),
    }))
  }
}

#[cfg(test)]
mod tests {
  use tonic::Code;

  use super::*;

  #[test]
  fn every_engine_error_maps_to_the_status_code_a_client_should_act_on() {
    let cases = [
      (EngineError::InvalidArgument("m".to_owned()), Code::InvalidArgument),
      (EngineError::NotFound("m".to_owned()), Code::NotFound),
      (
        EngineError::FailedPrecondition("m".to_owned()),
        Code::FailedPrecondition,
      ),
      (EngineError::Overloaded("m".to_owned()), Code::ResourceExhausted),
      (EngineError::Unavailable("m".to_owned()), Code::Unavailable),
      (EngineError::Internal("m".to_owned()), Code::Internal),
    ];
    for (error, code) in cases {
      let status = Status::from(error);
      assert_eq!((status.code(), status.message()), (code, "m"));
    }
  }

  /// One of each kind of reason, with numbers that are awkward on purpose.
  fn one_of_each_reason() -> Vec<Reason> {
    vec![
      Reason::BelowThreshold {
        similarity: 1.0 / 3.0,
        threshold: 0.9,
      },
      Reason::FieldBelowFloor {
        field: "name".to_owned(),
        similarity: 0.1 + 0.2,
        floor: 0.8,
      },
      Reason::AnchorDrift {
        similarity: 0.567,
        threshold: 0.85,
      },
      Reason::QuantityBeyondFactor {
        field: "weight".to_owned(),
        factor: Some(8.0),
        limit: 4.0,
      },
      Reason::QuantityIncomparable {
        field: "thc".to_owned(),
        previous_unit: "%".to_owned(),
        unit: "mg".to_owned(),
      },
      Reason::MeasureChanged,
    ]
  }

  #[test]
  fn every_reason_has_a_wire_code_of_its_own() {
    let reasons = one_of_each_reason();
    let mut wire: Vec<i32> = reasons.iter().map(|reason| reason_code(reason).into()).collect();
    wire.sort_unstable();
    wire.dedup();
    assert_eq!(wire.len(), reasons.len());
    assert!(!wire.contains(&i32::from(pb::ReasonCode::Unspecified)));
  }

  /// D17: the codes `calibrate --ndjson` writes are an interface. The API's enum is the same names, so a reason
  /// read from a response and one read from a calibration are joined without a table.
  #[test]
  fn a_wire_reason_code_is_named_after_the_code_calibrate_writes() {
    for reason in one_of_each_reason() {
      assert_eq!(
        reason_code(&reason).as_str_name(),
        format!("REASON_CODE_{}", reason.code().to_uppercase())
      );
    }
  }

  #[test]
  fn a_reason_detail_carries_the_subject_value_and_limit_of_its_reason() {
    let details: Vec<pb::ReasonDetail> = one_of_each_reason().iter().map(reason_detail).collect();
    let shape = |detail: &pb::ReasonDetail| {
      (
        detail.code(),
        detail.subject.clone(),
        detail.value.is_some(),
        detail.limit.is_some(),
      )
    };
    assert_eq!(
      details.iter().map(shape).collect::<Vec<_>>(),
      [
        (pb::ReasonCode::BelowThreshold, None, true, true),
        (pb::ReasonCode::FieldBelowFloor, Some("name".to_owned()), true, true),
        (pb::ReasonCode::AnchorDrift, None, true, true),
        (
          pb::ReasonCode::QuantityBeyondFactor,
          Some("weight".to_owned()),
          true,
          true
        ),
        (
          pb::ReasonCode::QuantityIncomparable,
          Some("thc".to_owned()),
          false,
          false
        ),
        (pb::ReasonCode::MeasureChanged, None, false, false),
      ]
    );
    assert_eq!((details[3].value, details[3].limit), (Some(8.0), Some(4.0)));

    // A quantity that moved between zero and a value above zero has no factor, and is still held to its limit.
    let from_zero = reason_detail(&Reason::QuantityBeyondFactor {
      field: "thc".to_owned(),
      factor: None,
      limit: 2.0,
    });
    assert_eq!((from_zero.value, from_zero.limit), (None, Some(2.0)));
  }

  #[test]
  #[allow(clippy::cast_possible_truncation, reason = "narrowing back is the point of the test")]
  fn a_similarity_on_the_wire_narrows_back_to_the_similarity_it_repeats() {
    for reason in one_of_each_reason() {
      let (similarity, limit) = match reason {
        Reason::BelowThreshold { similarity, threshold } | Reason::AnchorDrift { similarity, threshold } => {
          (similarity, threshold)
        }
        Reason::FieldBelowFloor { similarity, floor, .. } => (similarity, floor),
        _ => continue,
      };
      let detail = reason_detail(&reason);
      assert_eq!(detail.value.unwrap() as f32, similarity);
      assert_eq!(detail.limit.unwrap() as f32, limit);
    }
  }

  #[test]
  fn every_quantity_change_has_a_wire_value_of_its_own() {
    let changes = [
      QuantityChange::Unchanged,
      QuantityChange::Modified,
      QuantityChange::Incomparable,
      QuantityChange::Added,
      QuantityChange::Removed,
      QuantityChange::Restored,
    ];
    let mut wire: Vec<i32> = changes
      .into_iter()
      .map(|change| quantity_change(change).into())
      .collect();
    wire.sort_unstable();
    wire.dedup();
    assert_eq!(wire.len(), changes.len());
    assert!(!wire.contains(&i32::from(pb::QuantityChange::Unspecified)));

    // The five changes a text field can also have keep the numbers `FieldChange` gives them.
    for (quantity, field) in [
      (QuantityChange::Unchanged, FieldChange::Unchanged),
      (QuantityChange::Modified, FieldChange::Modified),
      (QuantityChange::Added, FieldChange::Added),
      (QuantityChange::Removed, FieldChange::Removed),
      (QuantityChange::Restored, FieldChange::Restored),
    ] {
      assert_eq!(
        i32::from(quantity_change(quantity)),
        i32::from(field_change(field)),
        "{quantity:?}"
      );
    }
  }

  #[test]
  fn an_unspecified_write_mode_enforces() {
    let decoded = |mode: pb::WriteMode| write_mode(mode.into()).unwrap();
    assert_eq!(decoded(pb::WriteMode::Unspecified), WriteMode::Enforce);
    assert_eq!(decoded(pb::WriteMode::Enforce), WriteMode::Enforce);
    assert_eq!(decoded(pb::WriteMode::DryRun), WriteMode::DryRun);
    assert_eq!(decoded(pb::WriteMode::Force), WriteMode::Force);
  }

  #[test]
  fn a_write_mode_this_server_does_not_know_is_refused_rather_than_enforced() {
    for unknown in [4, 99, -1] {
      let status = write_mode(unknown).unwrap_err();
      assert_eq!(status.code(), Code::InvalidArgument, "{unknown}");
    }
  }
}
