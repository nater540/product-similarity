//! The gRPC surface: protobuf in, [`Engine`] call, protobuf out.
//!
//! Nothing in this module decides anything. It translates messages, maps [`EngineError`] onto status codes a client
//! can act on, and logs one line per request. Keeping it that thin is what lets the engine's tests stand in for
//! tests of the service.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use similarity_core::matching::Decision;
use similarity_core::{FieldChange, FieldScore, Quantity, QuantityChange, QuantityScore, Reason};
use similarity_proto::v1 as pb;
use similarity_proto::v1::record_similarity_server::RecordSimilarity;
use similarity_store::VectorStore;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt as _};
use tonic::{Request, Response, Status, Streaming};
use tracing::Instrument as _;
use tracing::field::Empty;

use crate::engine::{
  Engine, EngineError, EvaluateInput, FindSimilarInput, ImportInput, Imported, MatchInput, Original, RecordInput,
  SimilarTo, Verdict, WriteMode,
};
use crate::import_stream::{self, Answer, PATIENCE};
use crate::metrics::Metrics;
use crate::worker::{self, TextEmbedder};

/// The tonic service wrapping an [`Engine`].
#[derive(Debug)]
pub struct GrpcService<S, E> {
  engine: Arc<Engine<S, E>>,
  metrics: Arc<Metrics>,
}

impl<S, E> GrpcService<S, E> {
  /// Wraps an engine, counting every request into `metrics`.
  pub fn new(engine: Arc<Engine<S, E>>, metrics: Arc<Metrics>) -> Self {
    Self { engine, metrics }
  }
}

/// The response half of an `Import` call.
type ImportStream = Pin<Box<dyn Stream<Item = Result<pb::ImportResponse, Status>> + Send>>;

impl From<EngineError> for Status {
  fn from(error: EngineError) -> Self {
    match error {
      EngineError::InvalidArgument(message) => Self::invalid_argument(message),
      EngineError::NotFound(message) => Self::not_found(message),
      EngineError::FailedPrecondition(message) => Self::failed_precondition(message),
      EngineError::Conflict(message) => Self::aborted(message),
      EngineError::Overloaded(message) => Self::resource_exhausted(message),
      EngineError::DeadlineExceeded(message) => Self::deadline_exceeded(message),
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

/// The header a gRPC client states its deadline in: how long from now it will wait.
const GRPC_TIMEOUT: &str = "grpc-timeout";

/// Reads a `grpc-timeout` value: up to eight digits and a unit, as the gRPC protocol over HTTP/2 defines it.
/// `None` for anything else, which is then no deadline, as it is for the transport.
fn grpc_timeout(value: &str) -> Option<Duration> {
  let unit = value.chars().last()?;
  let digits = value.strip_suffix(unit)?;
  if digits.is_empty() || digits.len() > 8 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
    return None;
  }
  let amount: u64 = digits.parse().ok()?;
  match unit {
    'H' => Some(Duration::from_secs(amount.saturating_mul(3600))),
    'M' => Some(Duration::from_secs(amount.saturating_mul(60))),
    'S' => Some(Duration::from_secs(amount)),
    'm' => Some(Duration::from_millis(amount)),
    'u' => Some(Duration::from_micros(amount)),
    'n' => Some(Duration::from_nanos(amount)),
    _ => None,
  }
}

/// When the caller of `request` stops waiting for it, if it said: `received` plus the timeout it sent.
///
/// The transport enforces the same timeout by dropping the handler when it runs out. This is for the one thing
/// that can be decided earlier than that: whether the request can be served in time at all (see
/// [`worker::by`]).
fn deadline<T>(request: &Request<T>, received: Instant) -> Option<Instant> {
  let timeout = grpc_timeout(request.metadata().get(GRPC_TIMEOUT)?.to_str().ok()?)?;
  received.checked_add(timeout)
}

fn record_input(record: pb::Record) -> RecordInput {
  RecordInput {
    id: record.id,
    brand_id: record.brand_id,
    fields: record.fields.into_iter().collect(),
    // Passed on exactly as sent. Whether a quantity is a usable measurement is the engine's to decide.
    quantities: quantities(record.quantities),
  }
}

/// Decodes the wire value of `EvaluateRequest.mode`.
///
/// A value this build does not know is refused rather than defaulted. The default writes, and a newer client
/// asking for a mode that does not exist yet (a second kind of preview, say) must not get a write instead.
fn quantities(quantities: std::collections::HashMap<String, pb::Quantity>) -> Vec<(String, Quantity)> {
  quantities
    .into_iter()
    .map(|(name, quantity)| (name, Quantity::range(quantity.min, quantity.max, quantity.unit)))
    .collect()
}

fn import_input(request: pb::ImportRequest) -> ImportInput {
  ImportInput {
    // A request without a record is a record without an id, and is answered as one.
    record: request.record.map(record_input).unwrap_or_default(),
    anchor: request.anchor.map(|original| Original {
      fields: original.fields.into_iter().collect(),
      quantities: quantities(original.quantities),
    }),
  }
}

fn import_response(answer: Answer) -> pb::ImportResponse {
  let (outcome, status_code, message) = match answer.outcome {
    Ok(Imported::Created) => (pb::ImportOutcome::Created, 0, String::new()),
    Ok(Imported::Skipped) => (pb::ImportOutcome::Exists, 0, String::new()),
    Err(error) => {
      let status = Status::from(error);
      // A status code is a small non-negative number.
      let code = u32::try_from(i32::from(status.code())).unwrap_or_default();
      (pb::ImportOutcome::Failed, code, status.message().to_owned())
    }
  };
  pb::ImportResponse {
    index: answer.index,
    id: answer.id,
    outcome: outcome.into(),
    status_code,
    message,
  }
}

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

/// Each call, from its request to its answer. The trait implementation below is these, each inside its span.
impl<S: VectorStore, E: TextEmbedder> GrpcService<S, E> {
  async fn evaluated(&self, request: Request<pb::EvaluateRequest>) -> Result<Response<pb::EvaluateResponse>, Status> {
    let started = Instant::now();
    let deadline = deadline(&request, started);
    let request = request.into_inner();
    let mode = write_mode(request.mode)?;
    let Some(record) = request.record else {
      return Err(Status::invalid_argument("`record` is required"));
    };
    let id = record.id.clone();

    let evaluated = self.engine.evaluate(EvaluateInput {
      record: record_input(record),
      mode,
      threshold: request.threshold,
    });
    let span = tracing::Span::current();
    span.record("record.id", id.as_deref().unwrap_or(""));
    span.record("mode", tracing::field::debug(mode));
    let output = worker::by(deadline, evaluated).await?;
    self.metrics.evaluated(output.verdict, mode);
    span.record("verdict", tracing::field::debug(output.verdict));
    span.record("stored", output.stored);
    // As a signed number: that is what a trace's attribute is, and an unsigned one is sent as text.
    span.record("revision", i64::try_from(output.revision).unwrap_or(i64::MAX));
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

  async fn found_similar(
    &self,
    request: Request<pb::FindSimilarRequest>,
  ) -> Result<Response<pb::FindSimilarResponse>, Status> {
    let started = Instant::now();
    let deadline = deadline(&request, started);
    let request = request.into_inner();
    let query = match request.query {
      Some(pb::find_similar_request::Query::Id(id)) => SimilarTo::Id(id),
      Some(pb::find_similar_request::Query::Record(record)) => SimilarTo::Record(record_input(record)),
      None => return Err(Status::invalid_argument("one of `id` or `record` is required")),
    };
    let found = self.engine.find_similar(FindSimilarInput {
      query,
      limit: usize::try_from(request.limit).unwrap_or(usize::MAX),
      brand_id: request.brand_id,
      min_similarity: request.min_similarity,
    });
    let neighbors = worker::by(deadline, found).await?;
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
          brand_id: neighbor.brand_id,
          similarity: neighbor.similarity,
        })
        .collect(),
    }))
  }

  async fn deleted(&self, request: Request<pb::DeleteRequest>) -> Result<Response<pb::DeleteResponse>, Status> {
    let started = Instant::now();
    let pb::DeleteRequest { id, brand_id } = request.into_inner();
    let span = tracing::Span::current();
    span.record("record.id", id.as_str());
    let deleted = self.engine.delete(id.clone(), brand_id).await?;
    span.record("deleted", deleted);
    tracing::info!(%id, deleted, elapsed_ms = milliseconds(started), "delete");
    Ok(Response::new(pb::DeleteResponse { deleted }))
  }

  /// Starts an import stream. The records are imported long after this has returned, by a task of its own, which
  /// is a span in the call's trace all the same.
  fn imported(&self, request: Request<Streaming<pb::ImportRequest>>) -> Response<ImportStream> {
    let started = Instant::now();
    let mut requests = request.into_inner();
    // Both channels hold one batch: a caller that sends faster than records are imported, or reads answers
    // slower than they are made, is slowed by the transport and not buffered without bound here.
    let batch = self.engine.import_batch();
    let (inputs, incoming) = mpsc::channel(batch);
    let (answers, answered) = mpsc::channel(batch);

    tokio::spawn(async move {
      loop {
        match requests.message().await {
          Ok(Some(message)) => {
            if inputs.send(import_input(message)).await.is_err() {
              break;
            }
          }
          Ok(None) => break,
          Err(status) => {
            // The caller broke the stream off. What was imported stays, and the caller knows what it saw.
            tracing::warn!(code = ?status.code(), "an import stream ended early");
            break;
          }
        }
      }
    });
    let engine = Arc::clone(&self.engine);
    let importing = tracing::debug_span!("import");
    let import = async move {
      let totals = import_stream::run(&engine, incoming, answers, PATIENCE).await;
      tracing::info!(
        records = totals.records,
        created = totals.created,
        existing = totals.existing,
        failed = totals.failed,
        batches = totals.batches,
        waits = totals.waits,
        elapsed_ms = milliseconds(started),
        "import"
      );
    };
    tokio::spawn(import.instrument(importing));

    let responses = ReceiverStream::new(answered).map(|answer| Ok(import_response(answer)));
    Response::new(Box::pin(responses))
  }

  async fn matched(&self, request: Request<pb::MatchRequest>) -> Result<Response<pb::MatchResponse>, Status> {
    let started = Instant::now();
    let deadline = deadline(&request, started);
    let request = request.into_inner();
    let Some(menu_item) = request.menu_item else {
      return Err(Status::invalid_argument("`menu_item` is required"));
    };
    let matched = self.engine.match_menu_item(MatchInput {
      menu_item: record_input(menu_item),
      limit: usize::try_from(request.limit).unwrap_or(usize::MAX),
      brand_id: request.brand_id,
      threshold: request.threshold,
    });
    let output = worker::by(deadline, matched).await?;
    tracing::Span::current().record("decision", tracing::field::debug(output.decision));
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
          brand_id: candidate.brand_id,
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

  fn described(&self) -> Response<pb::DescribeResponse> {
    let description = self.engine.describe();
    Response::new(pb::DescribeResponse {
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
    })
  }
}

/// The API: each method above inside a span named for the call, in which what the call waits for (its record's
/// lock, the store, the model) are spans of their own, and on which what came of it is written once that is
/// known (docs/decisions.md, D40). The transport's layer has already made a span for the request as it arrived;
/// these are inside that one. They are made at `debug`, so that a server nobody is tracing makes none.
#[tonic::async_trait]
impl<S: VectorStore, E: TextEmbedder> RecordSimilarity for GrpcService<S, E> {
  async fn evaluate(&self, request: Request<pb::EvaluateRequest>) -> Result<Response<pb::EvaluateResponse>, Status> {
    let span = tracing::debug_span!(
      "evaluate",
      record.id = Empty,
      mode = Empty,
      verdict = Empty,
      stored = Empty,
      revision = Empty
    );
    self.evaluated(request).instrument(span).await
  }

  async fn find_similar(
    &self,
    request: Request<pb::FindSimilarRequest>,
  ) -> Result<Response<pb::FindSimilarResponse>, Status> {
    self
      .found_similar(request)
      .instrument(tracing::debug_span!("find_similar"))
      .await
  }

  async fn delete(&self, request: Request<pb::DeleteRequest>) -> Result<Response<pb::DeleteResponse>, Status> {
    let span = tracing::debug_span!("delete", record.id = Empty, deleted = Empty);
    self.deleted(request).instrument(span).await
  }

  type ImportStream = ImportStream;

  async fn import(
    &self,
    request: Request<Streaming<pb::ImportRequest>>,
  ) -> Result<Response<Self::ImportStream>, Status> {
    Ok(self.imported(request))
  }

  async fn r#match(&self, request: Request<pb::MatchRequest>) -> Result<Response<pb::MatchResponse>, Status> {
    let span = tracing::debug_span!("match", decision = Empty);
    self.matched(request).instrument(span).await
  }

  async fn describe(&self, _: Request<pb::DescribeRequest>) -> Result<Response<pb::DescribeResponse>, Status> {
    Ok(self.described())
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
      (EngineError::Conflict("m".to_owned()), Code::Aborted),
      (EngineError::Overloaded("m".to_owned()), Code::ResourceExhausted),
      (EngineError::DeadlineExceeded("m".to_owned()), Code::DeadlineExceeded),
      (EngineError::Unavailable("m".to_owned()), Code::Unavailable),
      (EngineError::Internal("m".to_owned()), Code::Internal),
    ];
    for (error, code) in cases {
      let status = Status::from(error);
      assert_eq!((status.code(), status.message()), (code, "m"));
    }
  }

  #[test]
  fn a_grpc_timeout_is_digits_and_a_unit_and_anything_else_is_no_deadline() {
    let cases = [
      ("2H", Some(Duration::from_hours(2))),
      ("3M", Some(Duration::from_mins(3))),
      ("15S", Some(Duration::from_secs(15))),
      ("250m", Some(Duration::from_millis(250))),
      ("7u", Some(Duration::from_micros(7))),
      ("9n", Some(Duration::from_nanos(9))),
      ("0m", Some(Duration::ZERO)),
      ("99999999S", Some(Duration::from_secs(99_999_999))),
      // Nine digits are more than the protocol allows.
      ("123456789S", None),
      ("", None),
      ("S", None),
      ("15", None),
      ("15s", None),
      ("1.5S", None),
      ("-1S", None),
      ("+1S", None),
      (" 1S", None),
      ("1 S", None),
      ("1é", None),
    ];
    for (value, expected) in cases {
      assert_eq!(grpc_timeout(value), expected, "{value:?}");
    }
  }

  #[test]
  fn a_requests_deadline_is_when_it_was_received_plus_the_timeout_it_sent() {
    let received = Instant::now();
    let with = |value: &str| {
      let mut request = Request::new(());
      request.metadata_mut().insert(GRPC_TIMEOUT, value.parse().unwrap());
      request
    };
    assert_eq!(
      deadline(&Request::new(()), received),
      None,
      "a caller that says nothing waits forever"
    );
    assert_eq!(
      deadline(&with("250m"), received),
      Some(received + Duration::from_millis(250))
    );
    assert_eq!(deadline(&with("soon"), received), None);

    // As the transport hands a request over: the header it arrived with is in the metadata.
    let arrived = tonic::codegen::http::Request::builder()
      .header(GRPC_TIMEOUT, "3S")
      .body(())
      .unwrap();
    assert_eq!(
      deadline(&Request::from_http(arrived), received),
      Some(received + Duration::from_secs(3))
    );
  }

  /// The deadline a caller sends reaches the embedding thread, through the handler and the engine: a request
  /// whose time is already up is answered `DEADLINE_EXCEEDED` by the service itself, with nothing embedded.
  #[tokio::test]
  async fn a_request_that_arrives_with_no_time_left_is_refused_and_not_embedded() {
    use similarity_core::{FieldSpec, Policy, Schema};
    use similarity_store::MemoryStore;

    use crate::engine::store_layout;
    use crate::worker::EmbedWorker;

    let options = similarity_core::EmbedderOptions {
      max_tokens: 16,
      batch_size: 8,
      output_dim: None,
    };
    let counted = crate::telemetry::testing::paged();
    let embedder = similarity_core::testing::tiny_embedder(&options);
    let worker = EmbedWorker::spawn_measured(embedder, 4, 8, &counted.metrics()).unwrap();
    let schema = Schema::new(vec![FieldSpec {
      name: "name".to_owned(),
      weight: 1.0,
      required: true,
      reject_below: None,
    }])
    .unwrap();
    let (layout, dimension) = store_layout(&schema, &worker);
    let engine = Engine::new(
      MemoryStore::new(layout, dimension),
      worker.clone(),
      schema,
      Policy::new(0.75, None).unwrap(),
    );
    let service = GrpcService::new(Arc::new(engine), counted.metrics());
    let record = || pb::Record {
      fields: [("name".to_owned(), "blueberry gummies".to_owned())].into(),
      ..pb::Record::default()
    };
    let timed = |timeout: &str| {
      let mut request = Request::new(pb::EvaluateRequest {
        record: Some(record()),
        ..pb::EvaluateRequest::default()
      });
      request.metadata_mut().insert(GRPC_TIMEOUT, timeout.parse().unwrap());
      request
    };

    let refused = service.evaluate(timed("0m")).await.unwrap_err();
    assert_eq!(refused.code(), Code::DeadlineExceeded, "{refused:?}");
    let stats = worker.stats();
    assert_eq!((stats.too_late, stats.jobs, stats.texts), (1, 0, 0));

    let mut searching = Request::new(pb::FindSimilarRequest {
      query: Some(pb::find_similar_request::Query::Record(record())),
      ..pb::FindSimilarRequest::default()
    });
    searching.metadata_mut().insert(GRPC_TIMEOUT, "0m".parse().unwrap());
    let refused = service.find_similar(searching).await.unwrap_err();
    assert_eq!(refused.code(), Code::DeadlineExceeded, "{refused:?}");

    let mut matching = Request::new(pb::MatchRequest {
      menu_item: Some(record()),
      ..pb::MatchRequest::default()
    });
    matching.metadata_mut().insert(GRPC_TIMEOUT, "0m".parse().unwrap());
    let refused = service.r#match(matching).await.unwrap_err();
    assert_eq!(refused.code(), Code::DeadlineExceeded, "{refused:?}");
    assert_eq!(worker.stats().too_late, 3);

    // With time to spare it is served like any other.
    let answered = service.evaluate(timed("1M")).await.unwrap().into_inner();
    assert_eq!(answered.verdict(), pb::Verdict::Embedded);
    assert_eq!(worker.stats().jobs, 1);

    // The three that were refused are counted as refused for being too late, and only the one that was answered
    // has a verdict.
    let read = |series: &str| crate::telemetry::testing::counted(&counted, series);
    assert_eq!(
      read("record_similarity_embed_refused_total{reason=\"too_late\"}"),
      Some(3.0)
    );
    assert_eq!(
      read("record_similarity_embed_refused_total{reason=\"queue_full\"}"),
      Some(0.0)
    );
    let answered = "record_similarity_evaluations_total{mode=\"enforce\",verdict=\"embedded\"}";
    assert_eq!(read(answered), Some(1.0));
  }

  /// A call is a span, and its waits are spans inside it: for its record's lock, for the store (each operation
  /// by name), and for the model. They are made at `debug`, so a server that nobody is tracing makes none.
  /// (What a collector receives of them is tested in `telemetry.rs`.)
  #[test]
  fn a_request_and_its_waits_are_spans_for_whoever_collects_them() {
    use similarity_core::{FieldSpec, Policy, Schema};
    use similarity_store::MemoryStore;
    use tracing_subscriber::fmt::format::FmtSpan;

    use crate::engine::store_layout;
    use crate::telemetry::Traced;
    use crate::worker::EmbedWorker;

    #[derive(Clone, Default)]
    struct Lines(Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for Lines {
      fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if let Ok(mut lines) = self.0.lock() {
          lines.extend_from_slice(bytes);
        }
        Ok(bytes.len())
      }
      fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
      }
    }

    let spans_at = |level: tracing::Level| {
      let lines = Lines::default();
      let writer = lines.clone();
      let subscriber = tracing_subscriber::fmt()
        .with_max_level(level)
        .with_span_events(FmtSpan::NEW)
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
      tracing::subscriber::with_default(subscriber, || {
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        runtime.block_on(async {
          let options = similarity_core::EmbedderOptions {
            max_tokens: 16,
            batch_size: 8,
            output_dim: None,
          };
          let worker = EmbedWorker::spawn(similarity_core::testing::tiny_embedder(&options), 4, 8).unwrap();
          let schema = Schema::new(vec![FieldSpec {
            name: "name".to_owned(),
            weight: 1.0,
            required: true,
            reject_below: None,
          }])
          .unwrap();
          let (layout, dimension) = store_layout(&schema, &worker);
          let store = Traced(MemoryStore::new(layout, dimension));
          let engine = Engine::new(store, worker, schema, Policy::new(0.75, None).unwrap());
          let service = GrpcService::new(Arc::new(engine), Arc::new(Metrics::discarded()));
          let request = Request::new(pb::EvaluateRequest {
            record: Some(pb::Record {
              id: Some("p1".to_owned()),
              fields: [("name".to_owned(), "blueberry gummies".to_owned())].into(),
              ..pb::Record::default()
            }),
            ..pb::EvaluateRequest::default()
          });
          service.evaluate(request).await.unwrap();
        });
      });
      String::from_utf8(lines.0.lock().unwrap().clone()).unwrap()
    };

    let traced = spans_at(tracing::Level::DEBUG);
    // Each line is the path of spans to the one that was made, outermost first.
    let made = |inner: &str| {
      traced
        .lines()
        .filter(|line| line.ends_with(": new") && line.contains("evaluate") && line.contains(inner))
        .count()
    };
    assert_eq!(made("evaluate: "), 1, "{traced}");
    // Inside the call, in the order it waits.
    assert_eq!(made(":lock: "), 1, "{traced}");
    assert_eq!(made(":store{operation=\"get\"}: "), 1, "{traced}");
    assert_eq!(made(":embed: "), 1, "{traced}");
    // The write runs on a task of its own (D38), and is still the call's.
    assert_eq!(made(":store{operation=\"put_if\"}: "), 1, "{traced}");
    // By the time it writes, the call's span says which record it is about.
    assert!(
      traced
        .lines()
        .any(|line| line.contains("record.id=\"p1\"") && line.contains(":store{operation=\"put_if\"}")),
      "{traced}"
    );

    let quiet = spans_at(tracing::Level::INFO);
    assert!(!quiet.contains(": new"), "at info no span is made: {quiet}");
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
