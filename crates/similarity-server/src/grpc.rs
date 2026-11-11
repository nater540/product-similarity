//! The gRPC surface: protobuf in, [`Engine`] call, protobuf out.
//!
//! Nothing in this module decides anything. It translates messages, maps [`EngineError`] onto status codes a client
//! can act on, and logs one line per request. Keeping it that thin is what lets the engine's tests stand in for
//! tests of the service.

use std::sync::Arc;

use similarity_core::{FieldChange, Quantity, QuantityChange, QuantityScore};
use similarity_proto::v1 as pb;
use similarity_proto::v1::record_similarity_server::RecordSimilarity;
use similarity_store::VectorStore;
use tonic::{Request, Response, Status};

use crate::engine::{Engine, EngineError, EvaluateInput, FindSimilarInput, RecordInput, SimilarTo, Verdict, WriteMode};
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

fn width(value: usize) -> u32 {
  u32::try_from(value).unwrap_or(u32::MAX)
}

#[tonic::async_trait]
impl<S: VectorStore, E: TextEmbedder> RecordSimilarity for GrpcService<S, E> {
  async fn evaluate(&self, request: Request<pb::EvaluateRequest>) -> Result<Response<pb::EvaluateResponse>, Status> {
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
      similarity = output.similarity,
      anchor_similarity = output.anchor_similarity,
      stored = output.stored,
      revision = output.revision,
      measure_changed = output.measure_changed,
      "evaluate"
    );

    Ok(Response::new(pb::EvaluateResponse {
      verdict: verdict(output.verdict).into(),
      similarity: output.similarity,
      threshold: output.threshold,
      fields: output
        .fields
        .into_iter()
        .map(|score| pb::FieldScore {
          field: score.field,
          change: field_change(score.change).into(),
          similarity: score.similarity,
          weight: score.weight,
          below_floor: score.below_floor,
        })
        .collect(),
      anchor_similarity: output.anchor_similarity,
      stored: output.stored,
      reasons: output.reasons,
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
    tracing::info!(neighbors = neighbors.len(), "find_similar");

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
    let pb::DeleteRequest { id, tenant } = request.into_inner();
    let deleted = self.engine.delete(id.clone(), tenant).await?;
    tracing::info!(%id, deleted, "delete");
    Ok(Response::new(pb::DeleteResponse { deleted }))
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
