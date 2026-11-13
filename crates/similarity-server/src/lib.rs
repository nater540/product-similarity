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
//! - [`import_stream`]: the same over a streaming call, sharing the model with live requests.
//! - [`metrics`]: the few numbers the server keeps about itself.
//! - [`telemetry`]: where its traces and metrics go.
//! - [`offline`]: comparing and calibrating from the command line, without a store.
//! - [`shadow`]: a candidate policy judging live requests without deciding any.
//! - [`registry`]: the record of which model was put into service, and the gate that lets one in.

// Tests compute expected values with ordinary arithmetic and compare exact floats where exactness is the point.
#![cfg_attr(test, allow(clippy::arithmetic_side_effects, clippy::float_cmp))]

pub mod carve;
pub mod config;
pub mod engine;
pub mod fetch;
pub mod grpc;
pub mod import;
pub mod import_stream;
pub mod load;
pub mod metrics;
pub mod offline;
#[cfg(test)]
mod properties;
pub mod registry;
pub mod runs;
pub mod shadow;
pub mod telemetry;
pub mod worker;

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use similarity_proto::v1::record_similarity_server::RecordSimilarityServer;
use similarity_store::VectorStore;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tonic::transport::Server;
use tonic::transport::server::TcpIncoming;
use tonic_tracing_opentelemetry::middleware::filters::reject_healthcheck;
use tonic_tracing_opentelemetry::middleware::server::OtelGrpcLayer;

use crate::engine::Engine;
use crate::grpc::GrpcService;
use crate::metrics::Metrics;
use crate::worker::TextEmbedder;

/// How the server reports its health, and how it stops (docs/decisions.md, D37).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifecycle {
  /// How often the store is asked whether it still answers. The readiness the server reports is that old at
  /// most, plus the time one check takes; a check that takes longer than this counts as no answer.
  pub store_check_every: Duration,
  /// On [`Stop::Drain`], how long the server goes on serving after it has started reporting "not ready".
  pub drain_notice: Duration,
  /// After that, how long requests already in flight are given before the server stops anyway.
  pub drain_deadline: Duration,
}

impl Default for Lifecycle {
  fn default() -> Self {
    Self {
      store_check_every: Duration::from_secs(2),
      drain_notice: Duration::from_secs(5),
      drain_deadline: Duration::from_secs(20),
    }
  }
}

/// How a server is asked to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
  /// Stop taking requests now and finish the ones in flight. What a person at a keyboard means by Ctrl-C, and
  /// what a test means.
  Now,
  /// Leave the rotation first: report "not ready", go on serving for [`Lifecycle::drain_notice`] so that a load
  /// balancer can notice, and only then stop taking requests. What an orchestrator means by `SIGTERM`.
  Drain,
}

/// Serves the gRPC API on `listener` until `shutdown` resolves, then finishes in-flight requests and returns:
/// [`serve_until`] for a caller with one way of stopping and no settings of its own.
///
/// # Errors
///
/// As [`serve_until`].
pub async fn serve<S, E>(
  engine: Arc<Engine<S, E>>,
  listener: TcpListener,
  shutdown: impl Future<Output = ()> + Send,
) -> anyhow::Result<()>
where
  S: VectorStore,
  E: TextEmbedder,
{
  let (asked, stops) = mpsc::channel(1);
  let ask = async {
    shutdown.await;
    // The receiver is only gone if the server has already stopped by itself.
    let _ = asked.send(Stop::Now).await;
    // Kept, and never used again: a sender that is dropped would read as a second request to stop.
    std::future::pending::<()>().await;
  };
  // This caller asked for no endpoint, so nothing is kept.
  let metrics = Arc::new(Metrics::discarded());
  tokio::select! {
    served = serve_until(engine, listener, stops, Lifecycle::default(), metrics) => served,
    () = ask => Ok(()),
  }
}

/// Serves the gRPC API on `listener` until it is asked to stop on `stops`, and stops the way it was asked.
///
/// Alongside the API it serves the standard gRPC health service and server reflection (so `grpcurl` works
/// without the `.proto`). Health has two answers. The empty service name is *liveness*: the process is up and
/// its listener answers. `recordsimilarity.v1.RecordSimilarity` is *readiness*: a request sent now could be
/// served, because the embedding thread is alive and the store answered the last time it was asked
/// ([`Engine::check`], every [`Lifecycle::store_check_every`]). A server that is not ready is still live, and
/// still answers what it is sent: readiness is advice to whoever routes requests, not a gate.
///
/// Stopping, on [`Stop::Drain`]: not ready at once, serving for the notice, then no new requests and up to the
/// deadline for the ones in flight, and for any write a request left behind when its caller gave up
/// ([`Engine::settled`]): the process is not to end with a write half way to the store. A second request to stop
/// during either wait ends that wait. [`Stop::Now`] has no notice. A channel whose senders are all gone is a
/// request to stop now.
///
/// Taking a bound listener rather than an address lets a caller bind port 0 and learn the port before serving,
/// which is how the end-to-end tests run.
///
/// What the server counts about itself goes to `metrics` ([`metrics::Metrics`]), and every call but a health
/// check is a span for whoever is tracing ([`telemetry`]); whether either goes anywhere is the caller's to
/// arrange.
///
/// # Errors
///
/// Returns an error if the reflection descriptor cannot be loaded or the transport fails.
pub async fn serve_until<S, E>(
  engine: Arc<Engine<S, E>>,
  listener: TcpListener,
  mut stops: mpsc::Receiver<Stop>,
  lifecycle: Lifecycle,
  metrics: Arc<Metrics>,
) -> anyhow::Result<()>
where
  S: VectorStore,
  E: TextEmbedder,
{
  let (health_reporter, health_service) = tonic_health::server::health_reporter();
  // The first answer is found out before the listener can be asked for it, so that it is never a guess.
  let ready = report::<S, E>(&health_reporter, None, checked(&engine, lifecycle).await).await;
  let watch = tokio::spawn(watch(Arc::clone(&engine), health_reporter.clone(), lifecycle, ready));
  let reflection = tonic_reflection::server::Builder::configure()
    .register_encoded_file_descriptor_set(similarity_proto::FILE_DESCRIPTOR_SET)
    // Without this the health service still answers, but reflection-driven tools such as `grpcurl` cannot find it.
    .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
    .build_v1()
    .context("cannot build the gRPC reflection service")?;

  let (close, closed) = oneshot::channel::<()>();
  let writing = Arc::clone(&engine);
  let server = Server::builder()
    // A span for each call, in the caller's trace if the request carries one. Not for health checks: something
    // asks every few seconds, and a trace of each would bury the rest.
    .layer(OtelGrpcLayer::default().filter(reject_healthcheck))
    .add_service(health_service)
    .add_service(reflection)
    .add_service(RecordSimilarityServer::new(GrpcService::new(engine, metrics)))
    // `TcpIncoming` rather than a bare stream of connections: only it applies `TCP_NODELAY`, without which Nagle's
    // algorithm can hold a small unary response back for tens of milliseconds, several times the cost of
    // computing it.
    .serve_with_incoming_shutdown(TcpIncoming::from(listener).with_nodelay(Some(true)), async {
      let _ = closed.await;
    });
  tokio::pin!(server);
  let stopped_with_an_error = "the gRPC server stopped with an error";

  let stop = tokio::select! {
    outcome = &mut server => {
      watch.abort();
      return outcome.context(stopped_with_an_error);
    }
    stop = stops.recv() => stop.unwrap_or(Stop::Now),
  };
  // Out of the rotation before anything else, and for good: the watch must not put the server back in.
  watch.abort();
  let _ = watch.await;
  health_reporter
    .set_not_serving::<RecordSimilarityServer<GrpcService<S, E>>>()
    .await;
  let notice = match stop {
    Stop::Now => Duration::ZERO,
    Stop::Drain => lifecycle.drain_notice,
  };
  tracing::info!(
    notice_ms = millis(notice),
    deadline_ms = millis(lifecycle.drain_deadline),
    "stopping: no longer ready"
  );
  if !notice.is_zero() {
    tokio::select! {
      outcome = &mut server => return outcome.context(stopped_with_an_error),
      () = tokio::time::sleep(notice) => {}
      () = asked_again(&mut stops) => tracing::info!("asked to stop again: the notice is cut short"),
    }
  }

  // No new requests from here: the listener closes, and connections end as their requests do.
  let _ = close.send(());
  let finished = async {
    let outcome = (&mut server).await;
    // A request whose caller gave up while it was writing is gone, and its write is not: that is finished too.
    writing.settled().await;
    outcome
  };
  tokio::select! {
    outcome = finished => outcome.context(stopped_with_an_error),
    () = tokio::time::sleep(lifecycle.drain_deadline) => {
      // Returning is all there is to do: the connections are the runtime's, and end when the process does, which
      // for the binary is next.
      tracing::warn!(
        deadline_ms = millis(lifecycle.drain_deadline),
        "the drain deadline passed with requests or writes still in flight: stopping without them"
      );
      Ok(())
    }
    () = asked_again(&mut stops) => {
      tracing::warn!("asked to stop again with requests or writes still in flight: stopping without them");
      Ok(())
    }
  }
}

fn millis(duration: Duration) -> u64 {
  u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Resolves when the server is asked to stop once more, and never if nobody is left to ask.
async fn asked_again(stops: &mut mpsc::Receiver<Stop>) {
  if stops.recv().await.is_none() {
    std::future::pending::<()>().await;
  }
}

/// One look at whether a request could be served now: `None` if it could, or why not.
async fn checked<S, E>(engine: &Engine<S, E>, lifecycle: Lifecycle) -> Option<String>
where
  S: VectorStore,
  E: TextEmbedder,
{
  match tokio::time::timeout(lifecycle.store_check_every, engine.check()).await {
    Ok(Ok(())) => None,
    Ok(Err(error)) => Some(error.to_string()),
    Err(_) => Some(format!(
      "the store did not answer within {} ms",
      millis(lifecycle.store_check_every)
    )),
  }
}

/// Makes the readiness answer what was just found, and says so when that is a change. `was` is the answer before
/// (`None` before the first), and the answer now is returned.
async fn report<S, E>(reporter: &tonic_health::server::HealthReporter, was: Option<bool>, found: Option<String>) -> bool
where
  S: VectorStore,
  E: TextEmbedder,
{
  let ready = found.is_none();
  if was == Some(ready) {
    return ready;
  }
  match found {
    None => {
      reporter
        .set_serving::<RecordSimilarityServer<GrpcService<S, E>>>()
        .await;
      // The first answer of a server that starts well is not news; coming back is.
      if was.is_some() {
        tracing::info!("ready again");
      }
    }
    Some(reason) => {
      reporter
        .set_not_serving::<RecordSimilarityServer<GrpcService<S, E>>>()
        .await;
      tracing::warn!(reason, "not ready");
    }
  }
  ready
}

/// Keeps the readiness answer true for as long as the server serves.
async fn watch<S, E>(
  engine: Arc<Engine<S, E>>,
  reporter: tonic_health::server::HealthReporter,
  lifecycle: Lifecycle,
  mut ready: bool,
) where
  S: VectorStore,
  E: TextEmbedder,
{
  loop {
    tokio::time::sleep(lifecycle.store_check_every).await;
    ready = report::<S, E>(&reporter, Some(ready), checked(&engine, lifecycle).await).await;
  }
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
  use crate::engine::{Matching, store_layout};
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
    let engine = Arc::new(
      Engine::new(
        MemoryStore::new(layout, dimension),
        embedder,
        schema,
        Policy::new(0.75, None).unwrap(),
      )
      .with_matching(Matching {
        threshold: Some(0.6),
        overfetch: 3,
      }),
    );

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

  fn record(id: Option<&str>, brand_id: Option<&str>, fields: &[(&str, &str)]) -> pb::Record {
    pb::Record {
      id: id.map(str::to_owned),
      brand_id: brand_id.map(str::to_owned),
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
    assert_eq!(
      (description.match_threshold, description.match_overfetch),
      (Some(0.6), 3)
    );
  }

  /// The stream the import tests send: two new records, one that is stored, the same one under another brand,
  /// one with a field the schema lacks, one with no record at all, and one with its original version.
  fn import_requests() -> Vec<pb::ImportRequest> {
    let request = |record: pb::Record| pb::ImportRequest {
      record: Some(record),
      anchor: None,
    };
    vec![
      request(record(Some("a"), Some("brand-1"), &[("name", "sour diesel vape")])),
      request(record(
        Some("old"),
        Some("brand-1"),
        &[("name", "something else entirely")],
      )),
      request(record(
        Some("old"),
        Some("brand-2"),
        &[("name", "another brand's edit")],
      )),
      request(record(Some("b"), None, &[("strain", "indica")])),
      pb::ImportRequest::default(),
      pb::ImportRequest {
        anchor: Some(pb::Original {
          fields: HashMap::from([("name".to_owned(), "indoor flower".to_owned())]),
          quantities: HashMap::new(),
        }),
        ..request(record(Some("c"), Some("brand-2"), &[("name", "indoor flower eighth")]))
      },
    ]
  }

  async fn import(
    client: &mut RecordSimilarityClient<Channel>,
    requests: Vec<pb::ImportRequest>,
  ) -> Vec<pb::ImportResponse> {
    let mut answers = client.import(tokio_stream::iter(requests)).await.unwrap().into_inner();
    let mut seen = Vec::new();
    while let Some(answer) = answers.message().await.unwrap() {
      seen.push(answer);
    }
    seen
  }

  async fn with_one_stored() -> Running {
    let mut running = start().await;
    running
      .client
      .evaluate(evaluate(record(
        Some("old"),
        Some("brand-1"),
        &[("name", "blueberry gummies")],
      )))
      .await
      .unwrap();
    running
  }

  /// `Import` over the wire: a stream of records in, one answer for each out, in order. What is stored is left
  /// alone, a bad record fails by itself with the status code `Evaluate` would have given it, and an original
  /// version becomes the anchor.
  #[tokio::test]
  async fn records_are_created_over_a_stream_and_nothing_stored_is_replaced() {
    let mut running = with_one_stored().await;
    let client = &mut running.client;
    let seen = import(client, import_requests()).await;
    let summary: Vec<(u64, Option<&str>, pb::ImportOutcome, u32)> = seen
      .iter()
      .map(|answer| (answer.index, answer.id.as_deref(), answer.outcome(), answer.status_code))
      .collect();
    let (invalid, precondition) = (Code::InvalidArgument as u32, Code::FailedPrecondition as u32);
    assert_eq!(
      summary,
      [
        (0, Some("a"), pb::ImportOutcome::Created, 0),
        (1, Some("old"), pb::ImportOutcome::Exists, 0),
        (2, Some("old"), pb::ImportOutcome::Failed, precondition),
        (3, Some("b"), pb::ImportOutcome::Failed, invalid),
        (4, None, pb::ImportOutcome::Failed, invalid),
        (5, Some("c"), pb::ImportOutcome::Created, 0),
      ]
    );
    assert!(seen[0].message.is_empty());
    assert!(seen[3].message.contains("strain"), "{}", seen[3].message);
    assert!(seen[4].message.contains("id"), "{}", seen[4].message);

    // The stored record is what it was: the same text is still unchanged against it.
    let resaved = client
      .evaluate(evaluate(record(
        Some("old"),
        Some("brand-1"),
        &[("name", "blueberry gummies")],
      )))
      .await
      .unwrap()
      .into_inner();
    assert_eq!(
      (resaved.verdict(), resaved.stored, resaved.revision),
      (pb::Verdict::Accepted, false, 1)
    );
    assert_eq!(resaved.similarity, Some(1.0));
    // The original version is what drift is measured from.
    let drifted = client
      .evaluate(pb::EvaluateRequest {
        mode: pb::WriteMode::DryRun.into(),
        ..evaluate(record(Some("c"), Some("brand-2"), &[("name", "indoor flower eighth")]))
      })
      .await
      .unwrap()
      .into_inner();
    assert_eq!(drifted.similarity, Some(1.0));
    assert!(drifted.anchor_similarity.unwrap() < 0.99, "{drifted:?}");
  }

  /// A stream is safe to send twice: the second time creates nothing, replaces nothing, and fails what failed.
  #[tokio::test]
  async fn the_same_stream_sent_again_creates_nothing() {
    let mut running = with_one_stored().await;
    let client = &mut running.client;
    import(client, import_requests()).await;
    let outcomes: Vec<pb::ImportOutcome> = import(client, import_requests())
      .await
      .iter()
      .map(pb::ImportResponse::outcome)
      .collect();
    assert_eq!(
      outcomes,
      [
        pb::ImportOutcome::Exists,
        pb::ImportOutcome::Exists,
        pb::ImportOutcome::Failed,
        pb::ImportOutcome::Failed,
        pb::ImportOutcome::Failed,
        pb::ImportOutcome::Exists,
      ]
    );
    assert!(
      import(client, Vec::new()).await.is_empty(),
      "an empty stream is answered with nothing"
    );
  }

  /// `Match` over the wire: the menu item without an id, the candidates best first with both scores, the decision
  /// as an enum, the fields read with the stored record as the baseline, and the refusals as status codes.
  #[tokio::test]
  async fn a_menu_item_is_matched_over_grpc() {
    let mut running = start().await;
    let client = &mut running.client;
    for (id, brand_id, fields) in [
      (
        "gummies",
        "brand-1",
        [("name", "blueberry gummies 100mg"), ("category", "edibles gummies")].as_slice(),
      ),
      ("tin", "brand-2", [("name", "blueberry gummies 10pk tin")].as_slice()),
    ] {
      client
        .evaluate(evaluate(record(Some(id), Some(brand_id), fields)))
        .await
        .unwrap();
    }
    let menu_item = || record(None, None, &[("name", "blueberry gummies 100mg 10pk")]);

    let response = client
      .r#match(pb::MatchRequest {
        menu_item: Some(menu_item()),
        ..pb::MatchRequest::default()
      })
      .await
      .unwrap()
      .into_inner();
    assert_eq!(response.decision(), pb::MatchDecision::Matched);
    assert_eq!(response.threshold, Some(0.6));
    let ids: Vec<(&str, Option<&str>)> = response
      .candidates
      .iter()
      .map(|candidate| (candidate.id.as_str(), candidate.brand_id.as_deref()))
      .collect();
    assert_eq!(ids, [("gummies", Some("brand-1")), ("tin", Some("brand-2"))]);
    let best = &response.candidates[0];
    assert!(best.similarity > best.search_similarity, "{best:?}");
    assert!(best.similarity > response.candidates[1].similarity);
    let fields: Vec<(&str, pb::FieldChange, bool)> = best
      .fields
      .iter()
      .map(|field| (field.field.as_str(), field.change(), field.similarity.is_some()))
      .collect();
    assert_eq!(
      fields,
      [
        ("category", pb::FieldChange::Removed, false),
        ("name", pb::FieldChange::Modified, true)
      ]
    );
    assert!(best.fields.iter().all(|field| !field.below_floor));

    // The request's own threshold, its brand, and its limit.
    let narrowed = client
      .r#match(pb::MatchRequest {
        menu_item: Some(menu_item()),
        limit: 1,
        brand_id: Some("brand-2".to_owned()),
        threshold: Some(0.99),
      })
      .await
      .unwrap()
      .into_inner();
    assert_eq!(narrowed.decision(), pb::MatchDecision::NoMatch);
    assert_eq!(
      (narrowed.candidates.len(), narrowed.candidates[0].id.as_str()),
      (1, "tin")
    );
    assert_eq!(narrowed.threshold, Some(0.99));

    // Refused: no menu item, a menu item with an id, a limit too large, a threshold off the scale.
    let refused = [
      pb::MatchRequest::default(),
      pb::MatchRequest {
        menu_item: Some(record(Some("menu-7"), None, &[("name", "x")])),
        ..pb::MatchRequest::default()
      },
      pb::MatchRequest {
        menu_item: Some(menu_item()),
        limit: 101,
        ..pb::MatchRequest::default()
      },
      pb::MatchRequest {
        menu_item: Some(menu_item()),
        threshold: Some(2.0),
        ..pb::MatchRequest::default()
      },
    ];
    for request in refused {
      let status = client.r#match(request.clone()).await.unwrap_err();
      assert_eq!(status.code(), Code::InvalidArgument, "{request:?}");
    }
  }

  #[tokio::test]
  async fn records_are_searchable_and_deletable_over_grpc() {
    let mut running = start().await;
    let client = &mut running.client;
    for (id, brand_id, fields) in [
      ("gummies", "m1", &GUMMIES),
      ("gummies-100", "m2", &GUMMIES_EDITED),
      ("vape", "m1", &VAPE),
    ] {
      client
        .evaluate(evaluate(record(Some(id), Some(brand_id), fields)))
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
    assert_eq!(neighbors[0].brand_id.as_deref(), Some("m2"));
    assert!(neighbors[0].similarity > neighbors[1].similarity);

    let filtered = pb::FindSimilarRequest {
      brand_id: Some("m1".to_owned()),
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

    let delete = |brand_id: Option<&str>| pb::DeleteRequest {
      id: "vape".to_owned(),
      brand_id: brand_id.map(str::to_owned),
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

  // ---- Health and stopping (D37) ---------------------------------------------------------------------------

  use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

  use similarity_store::{Expected, Found, Neighbor, SearchQuery, StoreError, StoredRecord, Version};
  use tonic_health::pb::health_check_response::ServingStatus;

  use crate::worker::EmbedError;

  const LIVE: &str = "";
  const READY: &str = "recordsimilarity.v1.RecordSimilarity";

  /// A store whose readiness check a test can make fail, or never answer, and whose writes a test can hold on
  /// their way. Everything else is the memory store.
  struct Faulty {
    inner: MemoryStore,
    down: Arc<AtomicBool>,
    silent: Arc<AtomicBool>,
    /// While true, a write waits here: sent, and neither applied nor refused.
    writes_held: Arc<tokio::sync::watch::Sender<bool>>,
    /// How many writes have got that far.
    writes: Arc<AtomicUsize>,
  }

  impl Faulty {
    async fn let_through(&self) {
      self.writes.fetch_add(1, Ordering::Relaxed);
      let mut held = self.writes_held.subscribe();
      let _ = held.wait_for(|held| !*held).await;
    }
  }

  impl VectorStore for Faulty {
    async fn check(&self) -> Result<(), StoreError> {
      if self.silent.load(Ordering::Relaxed) {
        std::future::pending::<()>().await;
      }
      if self.down.load(Ordering::Relaxed) {
        return Err(StoreError::Unavailable("connection refused".to_owned()));
      }
      self.inner.check().await
    }

    async fn get(&self, id: &str) -> Result<Option<StoredRecord>, StoreError> {
      self.inner.get(id).await
    }

    async fn put(&self, record: &StoredRecord) -> Result<(), StoreError> {
      self.inner.put(record).await
    }

    async fn put_if(&self, record: &StoredRecord, expected: Expected) -> Result<bool, StoreError> {
      self.let_through().await;
      self.inner.put_if(record, expected).await
    }

    async fn delete(&self, id: &str) -> Result<bool, StoreError> {
      self.inner.delete(id).await
    }

    async fn delete_if(&self, id: &str, version: Version) -> Result<bool, StoreError> {
      self.let_through().await;
      self.inner.delete_if(id, version).await
    }

    async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>, StoreError> {
      self.inner.search(query).await
    }

    async fn search_vectors(&self, query: SearchQuery<'_>) -> Result<Vec<Found>, StoreError> {
      self.inner.search_vectors(query).await
    }
  }

  /// An embedder a test can declare dead, and can hold: a request that reaches it waits at the gate until the
  /// test lets it through, which is what a request in flight is.
  struct Gated {
    inner: WordBagEmbedder,
    alive: Arc<AtomicBool>,
    gate: Arc<tokio::sync::Semaphore>,
    waiting: Arc<AtomicUsize>,
  }

  impl TextEmbedder for Gated {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
      self.waiting.fetch_add(1, Ordering::Relaxed);
      let passed = self.gate.acquire().await;
      drop(passed);
      self.inner.embed(texts).await
    }

    fn dimension(&self) -> usize {
      self.inner.dimension()
    }

    fn fingerprint(&self) -> &str {
      self.inner.fingerprint()
    }

    fn alive(&self) -> bool {
      self.alive.load(Ordering::Relaxed)
    }
  }

  struct Lived {
    client: RecordSimilarityClient<Channel>,
    channel: Channel,
    stops: mpsc::Sender<Stop>,
    server: tokio::task::JoinHandle<anyhow::Result<()>>,
    down: Arc<AtomicBool>,
    silent: Arc<AtomicBool>,
    alive: Arc<AtomicBool>,
    gate: Arc<tokio::sync::Semaphore>,
    waiting: Arc<AtomicUsize>,
    writes_held: Arc<tokio::sync::watch::Sender<bool>>,
    writes: Arc<AtomicUsize>,
    counted: crate::telemetry::Telemetry,
  }

  /// A server over a store and an embedder the test controls. `open` says whether the embedder's gate starts
  /// open; `down` is the store's answer to the first check, the one made before the listener answers.
  async fn live(lifecycle: Lifecycle, open: bool, down: bool) -> Lived {
    let schema = Schema::new(vec![FieldSpec {
      name: "name".to_owned(),
      weight: 1.0,
      required: true,
      reject_below: None,
    }])
    .unwrap();
    let inner = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema, &inner);
    let (down, silent, alive) = (
      Arc::new(AtomicBool::new(down)),
      Arc::new(AtomicBool::new(false)),
      Arc::new(AtomicBool::new(true)),
    );
    let gate = Arc::new(tokio::sync::Semaphore::new(if open { 1_000_000 } else { 0 }));
    let waiting = Arc::new(AtomicUsize::new(0));
    let writes_held = Arc::new(tokio::sync::watch::Sender::new(false));
    let writes = Arc::new(AtomicUsize::new(0));
    let store = Faulty {
      inner: MemoryStore::new(layout, dimension),
      down: Arc::clone(&down),
      silent: Arc::clone(&silent),
      writes_held: Arc::clone(&writes_held),
      writes: Arc::clone(&writes),
    };
    let embedder = Gated {
      inner,
      alive: Arc::clone(&alive),
      gate: Arc::clone(&gate),
      waiting: Arc::clone(&waiting),
    };
    let engine = Arc::new(Engine::new(store, embedder, schema, Policy::new(0.75, None).unwrap()));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stops, asked) = mpsc::channel(4);
    let counted = crate::telemetry::testing::paged();
    let server = tokio::spawn(serve_until(engine, listener, asked, lifecycle, counted.metrics()));
    let channel = Channel::from_shared(format!("http://{address}"))
      .unwrap()
      .connect()
      .await
      .unwrap();
    Lived {
      client: RecordSimilarityClient::new(channel.clone()),
      channel,
      stops,
      server,
      down,
      silent,
      alive,
      gate,
      waiting,
      writes_held,
      writes,
      counted,
    }
  }

  /// Checks quickly, so that a test does not wait long for a change to be noticed; long waits for stopping, so
  /// that a test that sees the server stop knows it was not a wait running out.
  fn quick() -> Lifecycle {
    Lifecycle {
      store_check_every: Duration::from_millis(10),
      drain_notice: Duration::from_mins(10),
      drain_deadline: Duration::from_mins(10),
    }
  }

  async fn health(channel: &Channel, service: &str) -> ServingStatus {
    tonic_health::pb::health_client::HealthClient::new(channel.clone())
      .check(tonic_health::pb::HealthCheckRequest {
        service: service.to_owned(),
      })
      .await
      .unwrap()
      .into_inner()
      .status()
  }

  /// Waits for readiness to become `wanted`. Ten seconds is far more than a 10 ms check needs; it is a bound on
  /// a broken server, not an expectation of a working one.
  async fn becomes(channel: &Channel, wanted: ServingStatus) {
    for _ in 0..2000 {
      if health(channel, READY).await == wanted {
        return;
      }
      tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!(
      "readiness did not become {wanted:?}: it is {:?}",
      health(channel, READY).await
    );
  }

  fn named(name: &str) -> pb::EvaluateRequest {
    pb::EvaluateRequest {
      record: Some(pb::Record {
        id: Some("p1".to_owned()),
        fields: HashMap::from([("name".to_owned(), name.to_owned())]),
        ..pb::Record::default()
      }),
      ..pb::EvaluateRequest::default()
    }
  }

  /// Waits until a request is held at the embedder's gate. It takes milliseconds; three seconds is a bound on
  /// a broken server: a request that never gets there fails the test, and does not hang it.
  async fn held_at_the_gate(running: &Lived) {
    for _ in 0..1500 {
      if running.waiting.load(Ordering::Relaxed) > 0 {
        return;
      }
      tokio::time::sleep(Duration::from_millis(2)).await;
    }
    panic!("no request reached the embedder");
  }

  /// How long a test gives a server to stop once nothing keeps it. It takes milliseconds; this is a bound on a
  /// broken one, short enough that a server that never stops fails its test promptly.
  const STOPS_WITHIN: Duration = Duration::from_secs(8);

  /// The server's result once it has stopped, which it must do well within `within`.
  async fn stopped(server: tokio::task::JoinHandle<anyhow::Result<()>>, within: Duration) {
    tokio::time::timeout(within, server)
      .await
      .expect("the server did not stop")
      .unwrap()
      .unwrap();
  }

  /// Two questions, two answers. A store that stops answering makes the server not ready, and leaves it live:
  /// the process is fine, and restarting it would not bring the store back.
  #[tokio::test]
  async fn a_store_that_stops_answering_makes_the_server_not_ready_and_leaves_it_live() {
    let mut running = live(quick(), true, false).await;
    assert_eq!(health(&running.channel, LIVE).await, ServingStatus::Serving);
    assert_eq!(health(&running.channel, READY).await, ServingStatus::Serving);

    running.down.store(true, Ordering::Relaxed);
    becomes(&running.channel, ServingStatus::NotServing).await;
    assert_eq!(health(&running.channel, LIVE).await, ServingStatus::Serving);
    // Not ready is advice to whoever routes requests, not a gate: what arrives is still answered.
    running.client.describe(pb::DescribeRequest {}).await.unwrap();

    running.down.store(false, Ordering::Relaxed);
    becomes(&running.channel, ServingStatus::Serving).await;

    // A check that gets no answer at all is no answer: the watch does not wait on it for ever.
    running.silent.store(true, Ordering::Relaxed);
    becomes(&running.channel, ServingStatus::NotServing).await;
    running.silent.store(false, Ordering::Relaxed);
    becomes(&running.channel, ServingStatus::Serving).await;

    running.stops.send(Stop::Now).await.unwrap();
    drop((running.client, running.channel));
    stopped(running.server, STOPS_WITHIN).await;
  }

  /// The first answer is found out, not assumed: a server that starts with its store down has never said ready.
  #[tokio::test]
  async fn a_server_that_starts_without_its_store_is_not_ready_from_its_first_answer() {
    let slow_to_notice = Lifecycle {
      store_check_every: Duration::from_mins(10),
      ..quick()
    };
    let running = live(slow_to_notice, true, true).await;
    assert_eq!(health(&running.channel, READY).await, ServingStatus::NotServing);
    assert_eq!(health(&running.channel, LIVE).await, ServingStatus::Serving);
    drop(running.stops);
    drop((running.client, running.channel));
    // Nobody left to ask it to stop is a request to stop.
    stopped(running.server, STOPS_WITHIN).await;
  }

  #[tokio::test]
  async fn a_server_whose_embedding_thread_has_stopped_is_not_ready() {
    let running = live(quick(), true, false).await;
    assert_eq!(health(&running.channel, READY).await, ServingStatus::Serving);
    running.alive.store(false, Ordering::Relaxed);
    becomes(&running.channel, ServingStatus::NotServing).await;
    assert_eq!(health(&running.channel, LIVE).await, ServingStatus::Serving);
    running.stops.send(Stop::Now).await.unwrap();
    drop((running.client, running.channel));
    stopped(running.server, STOPS_WITHIN).await;
  }

  /// What an orchestrator's `SIGTERM` gets: out of the rotation at once, still serving while the notice lasts,
  /// and only then closed.
  #[tokio::test]
  async fn a_drain_leaves_the_rotation_first_and_serves_through_the_notice() {
    let lifecycle = Lifecycle {
      drain_notice: Duration::from_millis(400),
      ..quick()
    };
    let mut running = live(lifecycle, true, false).await;
    let asked = std::time::Instant::now();
    running.stops.send(Stop::Drain).await.unwrap();
    becomes(&running.channel, ServingStatus::NotServing).await;
    assert_eq!(health(&running.channel, LIVE).await, ServingStatus::Serving);
    // The store answers, and the server stays out all the same: a draining server does not come back.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(health(&running.channel, READY).await, ServingStatus::NotServing);
    let created = running.client.evaluate(named("blueberry gummies")).await.unwrap();
    assert_eq!(created.into_inner().verdict(), pb::Verdict::Created);

    drop((running.client, running.channel));
    stopped(running.server, STOPS_WITHIN).await;
    assert!(
      asked.elapsed() >= Duration::from_millis(400),
      "it stopped after {:?}, before the notice was over",
      asked.elapsed()
    );
  }

  /// A person at a keyboard is not made to wait for a load balancer that is not there, and an orchestrator that
  /// asks twice means now.
  #[tokio::test]
  async fn stop_now_gives_no_notice_and_asking_again_cuts_one_short() {
    let running = live(quick(), true, false).await;
    running.stops.send(Stop::Now).await.unwrap();
    drop((running.client, running.channel));
    stopped(running.server, STOPS_WITHIN).await;

    let running = live(quick(), true, false).await;
    running.stops.send(Stop::Drain).await.unwrap();
    becomes(&running.channel, ServingStatus::NotServing).await;
    assert!(!running.server.is_finished(), "the notice is ten minutes long");
    running.stops.send(Stop::Drain).await.unwrap();
    drop((running.client, running.channel));
    stopped(running.server, STOPS_WITHIN).await;
  }

  /// Stopping does not cut off a request that is being answered: the listener closes, and the server waits.
  #[tokio::test]
  async fn a_request_in_flight_is_answered_before_the_server_stops() {
    let running = live(quick(), false, false).await;
    let mut client = running.client.clone();
    let in_flight = tokio::spawn(async move { client.evaluate(named("blueberry gummies")).await });
    held_at_the_gate(&running).await;

    running.stops.send(Stop::Now).await.unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(!running.server.is_finished(), "it is waiting for the request");
    assert!(!in_flight.is_finished());

    running.gate.add_permits(1);
    let answered = in_flight.await.unwrap().unwrap();
    assert_eq!(answered.into_inner().verdict(), pb::Verdict::Created);
    drop((running.client, running.channel));
    stopped(running.server, STOPS_WITHIN).await;
  }

  /// A request that never finishes does not keep the server for ever: the deadline ends the wait, and so does
  /// being asked again.
  #[tokio::test]
  async fn the_deadline_or_a_second_request_ends_a_drain_that_requests_do_not() {
    for asked_twice in [false, true] {
      let lifecycle = Lifecycle {
        drain_deadline: if asked_twice {
          Duration::from_mins(10)
        } else {
          Duration::from_millis(200)
        },
        ..quick()
      };
      let running = live(lifecycle, false, false).await;
      let mut client = running.client.clone();
      let in_flight = tokio::spawn(async move { client.evaluate(named("blueberry gummies")).await });
      held_at_the_gate(&running).await;
      let asked = std::time::Instant::now();
      running.stops.send(Stop::Now).await.unwrap();
      if asked_twice {
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!running.server.is_finished(), "the deadline is ten minutes away");
        running.stops.send(Stop::Now).await.unwrap();
      }
      stopped(running.server, STOPS_WITHIN).await;
      if !asked_twice {
        assert!(asked.elapsed() >= Duration::from_millis(200), "{:?}", asked.elapsed());
      }
      // The request the server gave up on was never answered. Its connection belongs to the runtime, not to the
      // server, and ends when the process does: here the test ends it.
      assert!(!in_flight.is_finished(), "asked twice: {asked_twice}");
      in_flight.abort();
      drop((running.client, running.channel));
    }
  }

  // ---- Requests that go away (D38) --------------------------------------------------------------------------

  /// Runs `scene` on one thread, server included, and returns the warnings logged meanwhile.
  fn warnings(scene: impl Future<Output = ()>) -> Vec<serde_json::Value> {
    let captured = similarity_log::testing::capture("warn", || {
      tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(scene);
    });
    captured
      .logs
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect()
  }

  /// A request for `p1` whose caller waits `timeout` for it and no longer.
  fn impatient(name: &str, timeout: Duration) -> tonic::Request<pb::EvaluateRequest> {
    let mut request = tonic::Request::new(named(name));
    request.set_timeout(timeout);
    request
  }

  /// Sends a request whose write is held on its way to the store, and returns once its caller has given up:
  /// the write is in flight, and nobody is waiting for it.
  async fn given_up_mid_write(running: &Lived) {
    running.writes_held.send_replace(true);
    let mut client = running.client.clone();
    let timeout = Duration::from_millis(300);
    let call = tokio::spawn(async move { client.evaluate(impatient("blueberry gummies", timeout)).await });
    while running.writes.load(Ordering::Relaxed) == 0 {
      assert!(!call.is_finished(), "the caller gave up before the write was sent");
      tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let refused = call.await.unwrap().unwrap_err();
    // The transport's word for a timeout that ran out, on both sides of the connection.
    assert_eq!(refused.code(), tonic::Code::Cancelled, "{refused:?}");
    // The server's own timer was started a moment after the caller's.
    tokio::time::sleep(Duration::from_millis(100)).await;
  }

  fn gone(logged: &[serde_json::Value]) -> Vec<&serde_json::Value> {
    logged
      .iter()
      .filter(|line| line["message"] == "a write finished after the request that asked for it had gone")
      .collect()
  }

  /// The whole of it over a real connection. The caller's deadline runs out while its write is on its way; the
  /// transport drops the request; the write lands all the same, the log says so, and the caller's retry of the
  /// same record is told there is nothing to change.
  #[test]
  fn a_write_survives_the_deadline_of_the_request_that_asked_for_it() {
    let logged = warnings(async {
      let mut running = live(quick(), true, false).await;
      given_up_mid_write(&running).await;
      running.writes_held.send_replace(false);

      let retried = running
        .client
        .evaluate(named("blueberry gummies"))
        .await
        .unwrap()
        .into_inner();
      assert_eq!(
        (retried.verdict(), retried.stored, retried.revision),
        (pb::Verdict::Accepted, false, 1),
        "the record the caller was never told about is there"
      );
      running.stops.send(Stop::Now).await.unwrap();
      drop((running.client, running.channel));
      stopped(running.server, STOPS_WITHIN).await;
    });
    let gone = gone(&logged);
    assert_eq!(gone.len(), 1, "{logged:?}");
    assert_eq!((&gone[0]["call"], &gone[0]["id"]), (&"evaluate".into(), &"p1".into()));
    assert_eq!((&gone[0]["stored"], &gone[0]["revision"]), (&true.into(), &1.into()));
  }

  /// Stopping waits for a write that no request is waiting for any more. The connections are all idle, so
  /// without this the process would end with the write half way to the store.
  #[test]
  fn a_server_that_is_stopping_waits_for_a_write_whose_request_has_gone() {
    let logged = warnings(async {
      let running = live(quick(), true, false).await;
      given_up_mid_write(&running).await;

      running.stops.send(Stop::Now).await.unwrap();
      drop((running.client, running.channel));
      tokio::time::sleep(Duration::from_millis(150)).await;
      assert!(!running.server.is_finished(), "a write is still on its way");

      running.writes_held.send_replace(false);
      stopped(running.server, STOPS_WITHIN).await;
    });
    assert_eq!(gone(&logged).len(), 1, "{logged:?}");
  }

  /// A request in the queue for the model whose caller's deadline runs out is dropped by the transport, and
  /// costs nothing more: the server goes on answering.
  #[tokio::test]
  async fn a_request_whose_deadline_runs_out_before_it_writes_stores_nothing() {
    let mut running = live(quick(), false, false).await;
    let refused = running
      .client
      .evaluate(impatient("blueberry gummies", Duration::from_millis(100)))
      .await
      .unwrap_err();
    assert_eq!(refused.code(), tonic::Code::Cancelled, "{refused:?}");
    // The caller's timer ran out first; the server's was started a moment later, and until it runs out too the
    // request is still there, and would go on if the model were free now.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(running.writes.load(Ordering::Relaxed), 0);

    running.gate.add_permits(1_000_000);
    let created = running.client.evaluate(named("blueberry gummies")).await.unwrap();
    assert_eq!(
      created.into_inner().verdict(),
      pb::Verdict::Created,
      "nothing was stored before"
    );
    running.stops.send(Stop::Now).await.unwrap();
    drop((running.client, running.channel));
    stopped(running.server, STOPS_WITHIN).await;
  }

  // ---- What the server counts about itself (D39) -------------------------------------------------------------

  /// Through a real connection: a verdict is counted for each `Evaluate` that got one, under the write mode the
  /// request asked for, so that a preview's rejection is not counted with a refusal. A request that was refused
  /// before it was judged has no verdict to count.
  #[tokio::test]
  async fn verdicts_are_counted_by_the_write_mode_that_was_asked_for() {
    let running = live(quick(), true, false).await;
    let evaluations = |verdict: &str, mode: &str| {
      let series = format!("record_similarity_evaluations_total{{mode=\"{mode}\",verdict=\"{verdict}\"}}");
      crate::telemetry::testing::counted(&running.counted, &series)
    };
    assert_eq!(evaluations("accepted", "enforce"), Some(0.0), "there from the start");

    let mut client = running.client.clone();
    client.evaluate(named("blueberry gummies")).await.unwrap();
    client.evaluate(named("blueberry gummies 100mg")).await.unwrap();
    client.evaluate(named("sour diesel vape cartridge")).await.unwrap();
    let preview = pb::EvaluateRequest {
      mode: pb::WriteMode::DryRun.into(),
      ..named("sour diesel vape cartridge")
    };
    client.evaluate(preview).await.unwrap();
    let unknown_field = pb::EvaluateRequest {
      record: Some(pb::Record {
        fields: HashMap::from([("colour".to_owned(), "blue".to_owned())]),
        ..pb::Record::default()
      }),
      ..pb::EvaluateRequest::default()
    };
    client.evaluate(unknown_field).await.unwrap_err();

    assert_eq!(evaluations("created", "enforce"), Some(1.0));
    assert_eq!(evaluations("accepted", "enforce"), Some(1.0));
    assert_eq!(evaluations("rejected", "enforce"), Some(1.0));
    assert_eq!(evaluations("rejected", "dry_run"), Some(1.0));
    assert_eq!(evaluations("rejected", "force"), Some(0.0));
    assert_eq!(evaluations("embedded", "enforce"), Some(0.0));

    running.stops.send(Stop::Now).await.unwrap();
    drop((running.client, running.channel, client));
    stopped(running.server, STOPS_WITHIN).await;
  }
}
