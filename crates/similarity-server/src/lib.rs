//! The record-similarity gRPC server.
//!
//! The binary in `main.rs` is argument parsing and wiring; everything it wires lives here so that it can be tested
//! and embedded. Modules, in dependency order:
//!
//! - [`config`]: the TOML configuration file.
//! - [`worker`]: the dedicated thread that runs the model, and the [`worker::TextEmbedder`] seam the engine uses.
//! - [`engine`]: the service's behavior: validate, embed, compare, decide, write.
//! - [`grpc`]: protobuf translation and status codes around the engine.
//! - [`fetch`]: downloading and verifying a model directory.
//! - [`import`]: loading a whole catalog into the store, a batch at a time.
//! - [`offline`]: comparing and calibrating from the command line, without a store.

// Tests compute expected values with ordinary arithmetic and compare exact floats where exactness is the point.
#![cfg_attr(test, allow(clippy::arithmetic_side_effects, clippy::float_cmp))]

pub mod carve;
pub mod config;
pub mod engine;
pub mod fetch;
pub mod grpc;
pub mod import;
pub mod offline;
#[cfg(test)]
mod properties;
pub mod worker;

use std::future::Future;
use std::sync::Arc;

use anyhow::Context;
use similarity_proto::v1::record_similarity_server::RecordSimilarityServer;
use similarity_store::VectorStore;
use tokio::net::TcpListener;
use tonic::transport::Server;
use tonic::transport::server::TcpIncoming;

use crate::engine::Engine;
use crate::grpc::GrpcService;
use crate::worker::TextEmbedder;

/// Serves the gRPC API on `listener` until `shutdown` resolves, then finishes in-flight requests and returns.
///
/// Alongside the API it serves the standard gRPC health service (for load balancers and Kubernetes probes) and
/// server reflection (so `grpcurl` works without the `.proto`). Taking a bound listener rather than an address lets
/// a caller bind port 0 and learn the port before serving, which is how the end-to-end tests run.
///
/// # Errors
///
/// Returns an error if the reflection descriptor cannot be loaded or the transport fails.
pub async fn serve<S, E>(
  engine: Arc<Engine<S, E>>,
  listener: TcpListener,
  shutdown: impl Future<Output = ()> + Send,
) -> anyhow::Result<()>
where
  S: VectorStore,
  E: TextEmbedder,
{
  let (health_reporter, health_service) = tonic_health::server::health_reporter();
  health_reporter
    .set_serving::<RecordSimilarityServer<GrpcService<S, E>>>()
    .await;
  let reflection = tonic_reflection::server::Builder::configure()
    .register_encoded_file_descriptor_set(similarity_proto::FILE_DESCRIPTOR_SET)
    // Without this the health service still answers, but reflection-driven tools such as `grpcurl` cannot find it.
    .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
    .build_v1()
    .context("cannot build the gRPC reflection service")?;

  Server::builder()
    .add_service(health_service)
    .add_service(reflection)
    .add_service(RecordSimilarityServer::new(GrpcService::new(engine)))
    // `TcpIncoming` rather than a bare stream of connections: only it applies `TCP_NODELAY`, without which Nagle's
    // algorithm can hold a small unary response back for tens of milliseconds, several times the cost of
    // computing it.
    .serve_with_incoming_shutdown(TcpIncoming::from(listener).with_nodelay(Some(true)), shutdown)
    .await
    .context("the gRPC server stopped with an error")
}

#[cfg(test)]
mod tests {
  //! The service through a real socket: a tonic client talking to [`serve`] on an ephemeral port.
  //!
  //! The engine's rules are tested in `engine.rs`. These tests exist for what only the wire can get wrong: field
  //! presence (`optional`), enum mapping, the `oneof`, status codes, health, reflection, and shutdown.

  use std::collections::HashMap;

  use similarity_core::{FieldSpec, Policy, QuantitySpec, Schema};
  use similarity_proto::v1 as pb;
  use similarity_proto::v1::record_similarity_client::RecordSimilarityClient;
  use similarity_store::MemoryStore;
  use tokio::sync::oneshot;
  use tonic::Code;
  use tonic::transport::Channel;

  use super::*;
  use crate::engine::store_layout;
  use crate::worker::testing::WordBagEmbedder;

  struct Running {
    client: RecordSimilarityClient<Channel>,
    channel: Channel,
    stop: oneshot::Sender<()>,
    server: tokio::task::JoinHandle<anyhow::Result<()>>,
  }

  async fn start() -> Running {
    let field = |name: &str, weight: f32, required: bool, reject_below: Option<f32>| FieldSpec {
      name: name.to_owned(),
      weight,
      required,
      reject_below,
    };
    let quantity = |name: &str, reject_above_factor: Option<f64>| QuantitySpec {
      name: name.to_owned(),
      reject_above_factor,
      reject_incomparable: false,
    };
    let schema = Schema::new(vec![
      field("name", 4.0, true, Some(0.3)),
      field("category", 2.0, false, None),
    ])
    .unwrap()
    .with_quantities(vec![quantity("weight", Some(4.0)), quantity("thc", None)])
    .unwrap();
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema, &embedder);
    let engine = Arc::new(Engine::new(
      MemoryStore::new(layout, dimension),
      embedder,
      schema,
      Policy::new(0.75, None).unwrap(),
    ));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel::<()>();
    let server = tokio::spawn(serve(engine, listener, async {
      let _ = stopped.await;
    }));
    let channel = Channel::from_shared(format!("http://{address}"))
      .unwrap()
      .connect()
      .await
      .unwrap();
    Running {
      client: RecordSimilarityClient::new(channel.clone()),
      channel,
      stop,
      server,
    }
  }

  fn record(id: Option<&str>, tenant: Option<&str>, fields: &[(&str, &str)]) -> pb::Record {
    pb::Record {
      id: id.map(str::to_owned),
      tenant: tenant.map(str::to_owned),
      fields: fields
        .iter()
        .map(|(name, text)| ((*name).to_owned(), (*text).to_owned()))
        .collect::<HashMap<_, _>>(),
      quantities: HashMap::new(),
    }
  }

  /// The gummies under `p1`, with these quantities as `(name, min, max, unit)`.
  fn measured(quantities: &[(&str, f64, f64, &str)]) -> pb::EvaluateRequest {
    evaluate(pb::Record {
      quantities: quantities
        .iter()
        .map(|(name, min, max, unit)| {
          let quantity = pb::Quantity {
            min: *min,
            max: *max,
            unit: (*unit).to_owned(),
          };
          ((*name).to_owned(), quantity)
        })
        .collect(),
      ..record(Some("p1"), Some("m1"), &GUMMIES)
    })
  }

  fn evaluate(record: pb::Record) -> pb::EvaluateRequest {
    pb::EvaluateRequest {
      record: Some(record),
      ..pb::EvaluateRequest::default()
    }
  }

  const GUMMIES: [(&str, &str); 2] = [("name", "Blueberry Gummies"), ("category", "Edibles Gummies")];
  const GUMMIES_EDITED: [(&str, &str); 2] = [("name", "Blueberry Gummies 100mg"), ("category", "Edibles Gummies")];
  const VAPE: [(&str, &str); 2] = [("name", "Sour Diesel Vape Cartridge"), ("category", "Vape Pens")];

  #[tokio::test]
  async fn a_record_is_created_edited_and_protected_from_recycling_over_grpc() {
    let mut running = start().await;
    let client = &mut running.client;

    let created = client
      .evaluate(evaluate(record(Some("p1"), Some("m1"), &GUMMIES)))
      .await
      .unwrap()
      .into_inner();
    assert_eq!(created.verdict(), pb::Verdict::Created);
    assert!(created.stored);
    assert!(created.reasons.is_empty() && created.reason_details.is_empty());
    assert_eq!((created.similarity, created.revision), (None, 1));
    assert!(created.vector.is_empty(), "the vector is only returned on request");

    let edited = client
      .evaluate(evaluate(record(Some("p1"), Some("m1"), &GUMMIES_EDITED)))
      .await
      .unwrap()
      .into_inner();
    assert_eq!(edited.verdict(), pb::Verdict::Accepted);
    assert!(edited.stored && edited.similarity.unwrap() > 0.75);
    assert!(edited.reasons.is_empty() && edited.reason_details.is_empty());
    assert_eq!(edited.revision, 2);
    let name = edited.fields.iter().find(|field| field.field == "name").unwrap();
    assert_eq!(name.change(), pb::FieldChange::Modified);
    assert!(name.similarity.is_some() && !name.below_floor);
    let category = edited.fields.iter().find(|field| field.field == "category").unwrap();
    assert_eq!(
      (category.change(), category.similarity),
      (pb::FieldChange::Unchanged, Some(1.0))
    );
    assert!(edited.anchor_similarity.is_some());

    let recycled = client
      .evaluate(evaluate(record(Some("p1"), Some("m1"), &VAPE)))
      .await
      .unwrap()
      .into_inner();
    assert_eq!(recycled.verdict(), pb::Verdict::Rejected);
    assert!(!recycled.stored && !recycled.reasons.is_empty());
    assert_eq!(recycled.revision, 2);

    // One detail per sentence, in the same order, and each sentence is about what its detail says.
    assert_eq!(recycled.reasons.len(), recycled.reason_details.len());
    let below = &recycled.reason_details[0];
    assert_eq!(below.code(), pb::ReasonCode::BelowThreshold);
    assert!(recycled.reasons[0].contains("below the threshold"));
    assert_eq!(below.subject, None);
    #[allow(
      clippy::cast_possible_truncation,
      reason = "a similarity on the wire is an f32 widened"
    )]
    let narrowed = (below.value.unwrap() as f32, below.limit.unwrap() as f32);
    assert_eq!(narrowed, (recycled.similarity.unwrap(), recycled.threshold));
    assert!((recycled.threshold - 0.75).abs() < f32::EPSILON);
  }

  #[tokio::test]
  async fn quantities_cross_the_wire_in_both_directions() {
    let mut running = start().await;
    let client = &mut running.client;
    let flower = [("weight", 3.5, 3.5, "g"), ("thc", 18.0, 22.0, "%")];
    let created = client.evaluate(measured(&flower)).await.unwrap().into_inner();
    assert_eq!(created.verdict(), pb::Verdict::Created);
    assert!(created.quantities.is_empty() && !created.measure_changed);

    let find = |response: &pb::EvaluateResponse, name: &str| {
      response
        .quantities
        .iter()
        .find(|score| score.field == name)
        .cloned()
        .unwrap()
    };

    // The same weight in another unit, and a routine potency update: one unchanged, one a small factor.
    let batch = [("weight", 3500.0, 3500.0, "mg"), ("thc", 19.0, 23.0, "%")];
    let updated = client.evaluate(measured(&batch)).await.unwrap().into_inner();
    assert_eq!(updated.verdict(), pb::Verdict::Accepted);
    assert!(updated.stored, "a changed quantity is a change");
    let weight = find(&updated, "weight");
    assert_eq!(weight.change(), pb::QuantityChange::Unchanged);
    assert_eq!(weight.factor, Some(1.0));
    assert_eq!(
      (weight.previous_unit.as_deref(), weight.unit.as_deref()),
      (Some("g"), Some("mg"))
    );
    let thc = find(&updated, "thc");
    assert_eq!(thc.change(), pb::QuantityChange::Modified);
    assert!((thc.factor.unwrap() - 19.0 / 18.0).abs() < 1e-9);
    assert!(!thc.beyond_limit && !thc.incomparable && !thc.zero_crossing);

    // Eight times the weight is beyond the limit this server sets for it; a potency in milligrams is only reported,
    // because this server sets no limit for that.
    let resized = [("weight", 28.0, 28.0, "g"), ("thc", 100.0, 100.0, "mg")];
    let rejected = client.evaluate(measured(&resized)).await.unwrap().into_inner();
    assert_eq!(rejected.verdict(), pb::Verdict::Rejected);
    assert!(!rejected.stored && rejected.measure_changed);
    assert_eq!(rejected.reasons.len(), 1, "{:?}", rejected.reasons);
    let detail = &rejected.reason_details[0];
    assert_eq!(rejected.reason_details.len(), 1);
    assert_eq!(
      (detail.code(), detail.subject.as_deref()),
      (pb::ReasonCode::QuantityBeyondFactor, Some("weight"))
    );
    assert!((detail.value.unwrap() - 8.0).abs() < 1e-9 && detail.limit.is_some());
    assert!(
      rejected.reasons[0].contains("`weight`"),
      "the sentence beside it is about the same thing"
    );
    let weight = find(&rejected, "weight");
    assert!(weight.beyond_limit && (weight.factor.unwrap() - 8.0).abs() < 1e-9);
    let thc = find(&rejected, "thc");
    assert_eq!(thc.change(), pb::QuantityChange::Incomparable);
    assert!(thc.incomparable && thc.factor.is_none() && !thc.beyond_limit);

    // Presence: a removed quantity has no unit of its own, an added one no previous unit.
    let swapped = [("weight", 3.5, 3.5, "g")];
    let removed = client.evaluate(measured(&swapped)).await.unwrap().into_inner();
    let thc = find(&removed, "thc");
    assert_eq!(thc.change(), pb::QuantityChange::Removed);
    assert_eq!((thc.previous_unit.as_deref(), thc.unit), (Some("%"), None));
    assert_eq!(thc.factor, None);

    // A quantity the schema does not define, or one that is not a measurement, never gets as far as a verdict.
    for invalid in [
      [("height", 1.0, 1.0, "m")],
      [("weight", 5.0, 1.0, "g")],
      [("weight", 1.0, 1.0, "")],
    ] {
      let status = client.evaluate(measured(&invalid)).await.unwrap_err();
      assert_eq!(status.code(), Code::InvalidArgument, "{invalid:?}");
    }
  }

  #[tokio::test]
  async fn request_options_cross_the_wire() {
    let mut running = start().await;
    let client = &mut running.client;

    // No id: embedded only, and the vector comes back when asked for.
    let request = pb::EvaluateRequest {
      return_vector: true,
      ..evaluate(record(None, None, &GUMMIES))
    };
    let embedded = client.evaluate(request).await.unwrap().into_inner();
    assert_eq!(embedded.verdict(), pb::Verdict::Embedded);
    let description = client.describe(pb::DescribeRequest {}).await.unwrap().into_inner();
    assert_eq!(embedded.vector.len(), description.composite_dimension as usize);

    // Dry run: verdict without a write.
    let dry = pb::EvaluateRequest {
      mode: pb::WriteMode::DryRun.into(),
      ..evaluate(record(Some("p1"), None, &GUMMIES))
    };
    let previewed = client.evaluate(dry).await.unwrap().into_inner();
    assert_eq!((previewed.verdict(), previewed.stored), (pb::Verdict::Created, false));

    // Force: a rejected revision is written anyway.
    client
      .evaluate(evaluate(record(Some("p1"), None, &GUMMIES)))
      .await
      .unwrap();
    let force = pb::EvaluateRequest {
      mode: pb::WriteMode::Force.into(),
      ..evaluate(record(Some("p1"), None, &VAPE))
    };
    let forced = client.evaluate(force).await.unwrap().into_inner();
    assert_eq!((forced.verdict(), forced.stored), (pb::Verdict::Rejected, true));

    // Per-request threshold.
    let strict = pb::EvaluateRequest {
      threshold: Some(0.999),
      mode: pb::WriteMode::DryRun.into(),
      ..evaluate(record(
        Some("p1"),
        None,
        &[("name", "Sour Diesel Vape Cartridge 1g"), ("category", "Vape Pens")],
      ))
    };
    let judged = client.evaluate(strict).await.unwrap().into_inner();
    assert_eq!(judged.verdict(), pb::Verdict::Rejected);
    assert!((judged.threshold - 0.999).abs() < f32::EPSILON);
  }

  #[tokio::test]
  async fn describe_reports_the_schema_and_thresholds() {
    let mut running = start().await;
    let description = running
      .client
      .describe(pb::DescribeRequest {})
      .await
      .unwrap()
      .into_inner();
    assert_eq!(description.model_fingerprint, "word-bag-v1");
    assert_eq!(description.composite_dimension, description.dimension * 2);
    assert!((description.threshold - 0.75).abs() < f32::EPSILON);
    assert_eq!(description.anchor_threshold, None);
    let names: Vec<&str> = description.fields.iter().map(|field| field.name.as_str()).collect();
    assert_eq!(names, ["category", "name"]);
    let name = &description.fields[1];
    assert!(name.required && name.reject_below == Some(0.3));
    assert_eq!(description.fields[0].reject_below, None);

    let quantities: Vec<(&str, Option<f64>, bool)> = description
      .quantities
      .iter()
      .map(|quantity| {
        (
          quantity.name.as_str(),
          quantity.reject_above_factor,
          quantity.reject_incomparable,
        )
      })
      .collect();
    assert_eq!(quantities, [("thc", None, false), ("weight", Some(4.0), false)]);
    assert!(!description.reject_on_measure_change);
  }

  #[tokio::test]
  async fn records_are_searchable_and_deletable_over_grpc() {
    let mut running = start().await;
    let client = &mut running.client;
    for (id, tenant, fields) in [
      ("gummies", "m1", &GUMMIES),
      ("gummies-100", "m2", &GUMMIES_EDITED),
      ("vape", "m1", &VAPE),
    ] {
      client
        .evaluate(evaluate(record(Some(id), Some(tenant), fields)))
        .await
        .unwrap();
    }

    let by_id = pb::FindSimilarRequest {
      query: Some(pb::find_similar_request::Query::Id("gummies".to_owned())),
      ..pb::FindSimilarRequest::default()
    };
    let neighbors = client.find_similar(by_id.clone()).await.unwrap().into_inner().neighbors;
    let ids: Vec<&str> = neighbors.iter().map(|neighbor| neighbor.id.as_str()).collect();
    assert_eq!(ids, ["gummies-100", "vape"]);
    assert_eq!(neighbors[0].tenant.as_deref(), Some("m2"));
    assert!(neighbors[0].similarity > neighbors[1].similarity);

    let filtered = pb::FindSimilarRequest {
      tenant: Some("m1".to_owned()),
      limit: 5,
      min_similarity: Some(-1.0),
      ..by_id
    };
    let neighbors = client.find_similar(filtered).await.unwrap().into_inner().neighbors;
    assert_eq!(
      neighbors
        .iter()
        .map(|neighbor| neighbor.id.as_str())
        .collect::<Vec<_>>(),
      ["vape"]
    );

    let by_record = pb::FindSimilarRequest {
      query: Some(pb::find_similar_request::Query::Record(record(None, None, &VAPE))),
      limit: 1,
      ..pb::FindSimilarRequest::default()
    };
    let neighbors = client.find_similar(by_record).await.unwrap().into_inner().neighbors;
    assert_eq!(neighbors[0].id, "vape");

    let delete = |tenant: Option<&str>| pb::DeleteRequest {
      id: "vape".to_owned(),
      tenant: tenant.map(str::to_owned),
    };
    // The record belongs to `m1`, so a request that does not say so is refused.
    for wrong in [None, Some("m2")] {
      let status = client.delete(delete(wrong)).await.unwrap_err();
      assert_eq!(status.code(), Code::FailedPrecondition, "{wrong:?}");
    }
    assert!(client.delete(delete(Some("m1"))).await.unwrap().into_inner().deleted);
    assert!(!client.delete(delete(Some("m1"))).await.unwrap().into_inner().deleted);
  }

  #[tokio::test]
  async fn errors_arrive_as_the_right_status_codes() {
    let mut running = start().await;
    let client = &mut running.client;
    client
      .evaluate(evaluate(record(Some("p1"), Some("m1"), &GUMMIES)))
      .await
      .unwrap();

    let code = |result: Result<_, tonic::Status>| {
      result
        .map(|_: tonic::Response<pb::EvaluateResponse>| ())
        .unwrap_err()
        .code()
    };
    assert_eq!(
      code(client.evaluate(pb::EvaluateRequest::default()).await),
      Code::InvalidArgument
    );
    // A write mode from some future version of the API must not fall back to the default, which writes.
    let from_the_future = pb::EvaluateRequest {
      mode: 7,
      ..evaluate(record(Some("p9"), None, &GUMMIES))
    };
    assert_eq!(code(client.evaluate(from_the_future).await), Code::InvalidArgument);
    let find = pb::FindSimilarRequest {
      query: Some(pb::find_similar_request::Query::Id("p9".to_owned())),
      ..pb::FindSimilarRequest::default()
    };
    assert_eq!(
      client.find_similar(find).await.unwrap_err().code(),
      Code::NotFound,
      "and nothing was stored"
    );
    assert_eq!(
      code(
        client
          .evaluate(evaluate(record(Some("p2"), None, &[("colour", "blue")])))
          .await
      ),
      Code::InvalidArgument
    );
    assert_eq!(
      code(
        client
          .evaluate(evaluate(record(Some("p2"), None, &[("category", "Edibles")])))
          .await
      ),
      Code::InvalidArgument
    );
    assert_eq!(
      code(client.evaluate(evaluate(record(Some(""), None, &GUMMIES))).await),
      Code::InvalidArgument
    );
    assert_eq!(
      code(
        client
          .evaluate(evaluate(record(Some("p1"), Some("m2"), &GUMMIES)))
          .await
      ),
      Code::FailedPrecondition
    );

    let missing = pb::FindSimilarRequest {
      query: Some(pb::find_similar_request::Query::Id("nope".to_owned())),
      ..pb::FindSimilarRequest::default()
    };
    assert_eq!(client.find_similar(missing).await.unwrap_err().code(), Code::NotFound);
    assert_eq!(
      client
        .find_similar(pb::FindSimilarRequest::default())
        .await
        .unwrap_err()
        .code(),
      Code::InvalidArgument
    );
    assert_eq!(
      client.delete(pb::DeleteRequest::default()).await.unwrap_err().code(),
      Code::InvalidArgument
    );
  }

  #[tokio::test]
  async fn health_is_serving_and_shutdown_is_graceful() {
    let running = start().await;
    let mut health = tonic_health::pb::health_client::HealthClient::new(running.channel.clone());
    let service = "recordsimilarity.v1.RecordSimilarity".to_owned();
    let status = health
      .check(tonic_health::pb::HealthCheckRequest { service })
      .await
      .unwrap()
      .into_inner()
      .status();
    assert_eq!(status, tonic_health::pb::health_check_response::ServingStatus::Serving);

    running.stop.send(()).unwrap();
    drop((running.client, running.channel, health));
    running.server.await.unwrap().unwrap();
  }
}
