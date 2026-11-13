//! Where the server's traces and metrics go (docs/decisions.md, D39 and D40).
//!
//! All of it is OpenTelemetry's. A request is a trace: the tonic layer starts a span for each call
//! (`tonic-tracing-opentelemetry`), the server's own `tracing` spans inside it become its children
//! (`tracing-opentelemetry`), and the SDK sends them to a collector over gRPC. The few metrics
//! ([`crate::metrics`]) go to the same collector, and can be served as a Prometheus page as well. This module
//! builds that from the configuration and takes it down again; nothing else in the server knows a collector
//! exists.
//!
//! Nothing here is on unless it is asked for. With no collector and no page, no span is made, no metric is
//! kept, and no connection is opened.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use opentelemetry::KeyValue;
use opentelemetry::metrics::MeterProvider as _;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{MetricExporter, SpanExporter, WithExportConfig as _};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider};
use similarity_store::{Expected, Found, Neighbor, SearchQuery, StoreError, StoredRecord, VectorStore, Version};
use tokio::net::TcpListener;
use tracing::Instrument as _;
use tracing_subscriber::Layer as _;
use tracing_subscriber::filter::Targets;

use crate::metrics::Metrics;

/// The name this server gives itself to a collector: OpenTelemetry's `service.name`.
pub const SERVICE: &str = "record-similarity";

/// The path the metrics page is served at.
pub const PATH: &str = "/metrics";

/// The target of the spans the tonic layer makes, one for each call.
const CALLS: &str = "otel::tracing";

/// Where telemetry goes, as the configuration says it.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
  /// A collector to send traces and metrics to over gRPC (OTLP), as `http://host:4317`.
  pub otlp_endpoint: Option<String>,
  /// How often metrics are sent to it.
  pub metrics_every: Duration,
  /// Whether the metrics are also kept for a page ([`Telemetry::page`]).
  pub metrics_page: bool,
  /// The share of traces that are kept, from 0 to 1, for requests that do not arrive as part of a trace already.
  /// One that does is kept or not as its caller decided.
  pub trace_sample_ratio: f64,
}

impl Default for Settings {
  fn default() -> Self {
    Self {
      otlp_endpoint: None,
      metrics_every: Duration::from_mins(1),
      metrics_page: false,
      trace_sample_ratio: 1.0,
    }
  }
}

/// A server's telemetry while it runs.
#[derive(Debug)]
pub struct Telemetry {
  metrics: Arc<Metrics>,
  meters: Option<SdkMeterProvider>,
  tracers: Option<SdkTracerProvider>,
  page: Option<prometheus::Registry>,
}

impl Telemetry {
  /// No telemetry: no span, no metric, no connection.
  #[must_use]
  pub fn off() -> Self {
    Self {
      metrics: Arc::new(Metrics::discarded()),
      meters: None,
      tracers: None,
      page: None,
    }
  }

  /// Starts what `settings` asks for, and returns it with the layer that turns the process's spans into traces,
  /// if there is a collector to send them to. The layer has to be part of the one subscriber the process has
  /// (`similarity_log::Init::tracing`), so this is called before logging is set up.
  ///
  /// With a collector, this has to be called inside a tokio runtime: the connection to it lives there. A
  /// collector that is not there is not an error: what there is to send is sent when it is.
  ///
  /// # Errors
  ///
  /// Returns an error if the collector's address is not one, or an exporter cannot be built.
  pub fn start(settings: &Settings) -> anyhow::Result<(Self, Option<similarity_log::Tracing>)> {
    let resource = Resource::builder()
      .with_service_name(SERVICE)
      .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
      .build();
    let mut telemetry = Self::off();

    let mut meters = SdkMeterProvider::builder().with_resource(resource.clone());
    if settings.metrics_page {
      let registry = prometheus::Registry::new();
      let exporter = opentelemetry_prometheus::exporter()
        .with_registry(registry.clone())
        // The names are written whole in `metrics.rs`, so that a page and a collector show the same ones.
        .without_counter_suffixes()
        .without_units()
        .scope_info_enabled(false)
        .without_target_info()
        .build()
        .context("cannot keep metrics for a page")?;
      meters = meters.with_reader(exporter);
      telemetry.page = Some(registry);
    }
    if let Some(endpoint) = &settings.otlp_endpoint {
      let exporter = MetricExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint.clone())
        .build()
        .with_context(|| format!("cannot send metrics to `{endpoint}`"))?;
      let reader = PeriodicReader::builder(exporter)
        .with_interval(settings.metrics_every)
        .build();
      meters = meters.with_reader(reader);
    }
    if settings.metrics_page || settings.otlp_endpoint.is_some() {
      let meters = meters.build();
      telemetry.metrics = Arc::new(Metrics::new(&meters.meter(SERVICE)));
      telemetry.meters = Some(meters);
    }

    let Some(endpoint) = &settings.otlp_endpoint else {
      return Ok((telemetry, None));
    };
    let exporter = SpanExporter::builder()
      .with_tonic()
      .with_endpoint(endpoint.clone())
      .build()
      .with_context(|| format!("cannot send traces to `{endpoint}`"))?;
    let tracers = SdkTracerProvider::builder()
      .with_resource(resource)
      // A request that arrives inside a trace is kept or not as its caller decided; one that starts a trace
      // here is kept with this probability.
      .with_sampler(Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
        settings.trace_sample_ratio,
      ))))
      .with_batch_exporter(exporter)
      .build();
    // How a caller's trace is recognised in a request's headers: the W3C `traceparent`.
    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
    let layer = tracing_opentelemetry::layer()
      .with_tracer(tracers.tracer(SERVICE))
      // The spans this layer wants are made for it alone: each call's, at whatever level the tonic layer makes
      // them, and this server's own, which are at `debug` so that a server nobody is tracing makes none.
      .with_filter(
        Targets::new()
          .with_target(CALLS, tracing::Level::TRACE)
          .with_target("similarity_server", tracing::Level::DEBUG),
      )
      .boxed();
    telemetry.tracers = Some(tracers);
    Ok((telemetry, Some(layer)))
  }

  /// Where the server counts.
  #[must_use]
  pub fn metrics(&self) -> Arc<Metrics> {
    Arc::clone(&self.metrics)
  }

  /// The metrics as a page in the Prometheus text format, if they are kept for one.
  #[must_use]
  pub fn page(&self) -> Option<String> {
    let registry = self.page.as_ref()?;
    prometheus::TextEncoder::new().encode_to_string(&registry.gather()).ok()
  }

  /// Sends what there is to the collector now, and says whether it took it. `true` with nothing to send to.
  /// Blocks until the collector has answered or timed out: for tests.
  #[must_use]
  pub fn flush(&self) -> bool {
    let traces = self
      .tracers
      .as_ref()
      .is_none_or(|tracers| tracers.force_flush().is_ok());
    let metrics = self.meters.as_ref().is_none_or(|meters| meters.force_flush().is_ok());
    traces && metrics
  }

  /// Sends what is left and stops sending. Blocks like [`Self::flush`], so a server calls it off the runtime's
  /// own threads, last thing.
  pub fn shutdown(&self) {
    if let Some(tracers) = &self.tracers
      && let Err(error) = tracers.shutdown()
    {
      tracing::warn!(%error, "the last traces were not sent to the collector");
    }
    if let Some(meters) = &self.meters
      && let Err(error) = meters.shutdown()
    {
      tracing::warn!(%error, "the last metrics were not sent to the collector");
    }
  }
}

/// A store whose every operation is a span, named for the operation: the store's part of a request's trace. It
/// changes nothing about what the store does, and no store has to know it is watched.
#[derive(Debug)]
pub struct Traced<S>(pub S);

impl<S: VectorStore> VectorStore for Traced<S> {
  async fn check(&self) -> Result<(), StoreError> {
    // Asked every few seconds by the server itself, and part of no request: not worth a span.
    self.0.check().await
  }

  async fn get(&self, id: &str) -> Result<Option<StoredRecord>, StoreError> {
    let span = tracing::debug_span!("store", operation = "get");
    self.0.get(id).instrument(span).await
  }

  async fn put(&self, record: &StoredRecord) -> Result<(), StoreError> {
    let span = tracing::debug_span!("store", operation = "put");
    self.0.put(record).instrument(span).await
  }

  async fn put_if(&self, record: &StoredRecord, expected: Expected) -> Result<bool, StoreError> {
    let span = tracing::debug_span!("store", operation = "put_if");
    self.0.put_if(record, expected).instrument(span).await
  }

  async fn delete(&self, id: &str) -> Result<bool, StoreError> {
    let span = tracing::debug_span!("store", operation = "delete");
    self.0.delete(id).instrument(span).await
  }

  async fn delete_if(&self, id: &str, version: Version) -> Result<bool, StoreError> {
    let span = tracing::debug_span!("store", operation = "delete_if");
    self.0.delete_if(id, version).instrument(span).await
  }

  async fn search(&self, query: SearchQuery<'_>) -> Result<Vec<Neighbor>, StoreError> {
    let span = tracing::debug_span!("store", operation = "search");
    self.0.search(query).instrument(span).await
  }

  async fn search_vectors(&self, query: SearchQuery<'_>) -> Result<Vec<Found>, StoreError> {
    let span = tracing::debug_span!("store", operation = "search_vectors");
    self.0.search_vectors(query).instrument(span).await
  }
}

/// Serves what `render` returns at [`PATH`] over HTTP on `listener`, until `stopped` resolves.
///
/// One path, `GET` only, and nothing else. It is plain HTTP with no authentication, like the gRPC port, and
/// belongs on the same private network.
///
/// # Errors
///
/// Returns an error if the listener fails.
pub async fn serve_page(
  listener: TcpListener,
  render: impl Fn() -> String + Send + Sync + 'static,
  stopped: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
  let render = Arc::new(render);
  let page = axum::routing::get(move || {
    let render = Arc::clone(&render);
    async move {
      let content_type = (axum::http::header::CONTENT_TYPE, prometheus::TEXT_FORMAT);
      ([content_type], render())
    }
  });
  axum::serve(listener, axum::Router::new().route(PATH, page))
    .with_graceful_shutdown(stopped)
    .await
    .context("the metrics page stopped with an error")
}

/// What the tests of other modules need to read a server's metrics.
#[cfg(test)]
pub(crate) mod testing {
  use super::{Settings, Telemetry};

  /// Telemetry that keeps its metrics for a page and sends nothing anywhere.
  pub(crate) fn paged() -> Telemetry {
    let settings = Settings {
      metrics_page: true,
      ..Settings::default()
    };
    Telemetry::start(&settings).unwrap().0
  }

  /// The value of the one series that is exactly `series`, in what `telemetry` has counted: `None` if it is not
  /// there.
  pub(crate) fn counted(telemetry: &Telemetry, series: &str) -> Option<f64> {
    let page = telemetry.page()?;
    page
      .lines()
      .find_map(|line| line.strip_prefix(series)?.strip_prefix(' ')?.parse().ok())
  }
}

#[cfg(test)]
mod tests {
  use std::collections::HashMap;
  use std::sync::atomic::{AtomicU64, Ordering};

  use fake_opentelemetry_collector::{ExportedSpan, FakeCollectorServer};
  use similarity_core::{FieldSpec, Policy, Schema};
  use similarity_proto::v1 as pb;
  use similarity_proto::v1::record_similarity_client::RecordSimilarityClient;
  use similarity_store::MemoryStore;
  use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
  use tokio::sync::mpsc;
  use tracing_subscriber::layer::SubscriberExt as _;

  use super::*;
  use crate::engine::{Engine, store_layout};
  use crate::worker::EmbedWorker;
  use crate::{Lifecycle, Stop, serve_until};

  /// With nothing asked for there is nothing: no layer to make spans for, no page, nothing to send, and metrics
  /// that keep no count.
  #[test]
  fn nothing_is_on_until_it_is_asked_for() {
    let (telemetry, tracing) = Telemetry::start(&Settings::default()).unwrap();
    assert!(tracing.is_none());
    assert_eq!(telemetry.page(), None);
    assert!(telemetry.flush(), "nothing to send is nothing that failed");
    telemetry.metrics().abandoned();
    telemetry.shutdown();

    // A page alone keeps the metrics and still traces nothing: there is nowhere to send a trace.
    let paged = Settings {
      metrics_page: true,
      ..Settings::default()
    };
    let (telemetry, tracing) = Telemetry::start(&paged).unwrap();
    assert!(tracing.is_none());
    telemetry.metrics().abandoned();
    let page = telemetry.page().unwrap();
    assert!(page.contains("record_similarity_embed_abandoned_total 1"), "{page}");
  }

  /// What a scraper does: a plain HTTP `GET`, read to the end.
  async fn get(address: std::net::SocketAddr, path: &str) -> (String, String) {
    let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
    let request = format!("GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut answer = String::new();
    stream.read_to_string(&mut answer).await.unwrap();
    let (head, body) = answer.split_once("\r\n\r\n").unwrap();
    (head.to_lowercase(), body.to_owned())
  }

  /// The page is one path. It answers a plain `GET` with what it is given to say, as the Prometheus text format,
  /// each time anew, and knows no other path.
  #[tokio::test]
  async fn the_page_is_served_at_one_path_and_nothing_anywhere_else() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let scrapes = Arc::new(AtomicU64::new(0));
    let counted = Arc::clone(&scrapes);
    let render = move || format!("scrapes {}\n", counted.fetch_add(1, Ordering::Relaxed));
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(serve_page(listener, render, async {
      let _ = stopped.await;
    }));

    let (head, body) = get(address, PATH).await;
    assert!(head.starts_with("http/1.1 200"), "{head}");
    assert!(head.contains("content-type: text/plain; version=0.0.4"), "{head}");
    assert!(body.contains("scrapes 0"), "{body}");
    let (_, body) = get(address, PATH).await;
    assert!(body.contains("scrapes 1"), "asked again, answered again: {body}");

    for elsewhere in ["/", "/metrics/", "/health", "/metricsx"] {
      let (head, _) = get(address, elsewhere).await;
      assert!(head.starts_with("http/1.1 404"), "{elsewhere}: {head}");
    }
    assert_eq!(
      scrapes.load(Ordering::Relaxed),
      2,
      "a path that is not there renders nothing"
    );

    stop.send(()).unwrap();
    server.await.unwrap().unwrap();
  }

  const TRACE: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
  const CALLER: &str = "00f067aa0ba902b7";

  /// One `Evaluate` for the record `p1`, from a caller that is already inside a trace.
  fn traced_request(name: &str) -> tonic::Request<pb::EvaluateRequest> {
    let mut request = tonic::Request::new(pb::EvaluateRequest {
      record: Some(pb::Record {
        id: Some("p1".to_owned()),
        fields: HashMap::from([("name".to_owned(), name.to_owned())]),
        ..pb::Record::default()
      }),
      ..pb::EvaluateRequest::default()
    });
    let caller = format!("00-{TRACE}-{CALLER}-01");
    request.metadata_mut().insert("traceparent", caller.parse().unwrap());
    request
  }

  /// What a collector received from a server that served one request, and what the server's page said then.
  struct Received {
    spans: Vec<ExportedSpan>,
    metrics: Vec<serde_json::Value>,
    page: String,
  }

  /// One `Evaluate` over a real connection to a server whose telemetry goes, over gRPC, to a collector that
  /// keeps what it is sent.
  fn one_request_seen_by_a_collector() -> Received {
    // One thread for everything, so that the spans every task makes are seen by this test's own subscriber.
    let runtime = tokio::runtime::Builder::new_current_thread()
      .enable_all()
      .build()
      .unwrap();
    let mut collector = runtime.block_on(FakeCollectorServer::start()).unwrap();
    let settings = Settings {
      otlp_endpoint: Some(collector.endpoint()),
      // Never, as far as this test goes: it sends when it is told to.
      metrics_every: Duration::from_hours(1),
      metrics_page: true,
      trace_sample_ratio: 1.0,
    };
    let (telemetry, tracing) = {
      let _on_the_runtime = runtime.enter();
      Telemetry::start(&settings).unwrap()
    };
    let telemetry = Arc::new(telemetry);
    let subscriber = tracing_subscriber::registry().with(tracing.expect("a collector means a tracing layer"));

    let (spans, metrics) = tracing::subscriber::with_default(subscriber, || {
      runtime.block_on(async {
        let options = similarity_core::EmbedderOptions {
          max_tokens: 16,
          batch_size: 8,
          output_dim: None,
        };
        let embedder = similarity_core::testing::tiny_embedder(&options);
        let worker = EmbedWorker::spawn_measured(embedder, 4, 8, &telemetry.metrics()).unwrap();
        let schema = Schema::new(vec![FieldSpec {
          name: "name".to_owned(),
          weight: 1.0,
          required: true,
          reject_below: None,
        }])
        .unwrap();
        let (layout, dimension) = store_layout(&schema, &worker);
        let store = Traced(MemoryStore::new(layout, dimension));
        let engine =
          Engine::new(store, worker, schema, Policy::new(0.75, None).unwrap()).with_metrics(telemetry.metrics());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (stops, asked) = mpsc::channel(1);
        let serving = serve_until(
          Arc::new(engine),
          listener,
          asked,
          Lifecycle::default(),
          telemetry.metrics(),
        );
        let server = tokio::spawn(serving);
        let mut client = RecordSimilarityClient::connect(format!("http://{address}"))
          .await
          .unwrap();

        // A health check first, as something asks for every few seconds: it is answered, and is no part of
        // what the collector is sent.
        let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
          .unwrap()
          .connect()
          .await
          .unwrap();
        let health = tonic_health::pb::health_client::HealthClient::new(channel)
          .check(tonic_health::pb::HealthCheckRequest { service: String::new() })
          .await
          .unwrap();
        assert_eq!(
          health.into_inner().status(),
          tonic_health::pb::health_check_response::ServingStatus::Serving
        );
        let created = client.evaluate(traced_request("blueberry gummies")).await.unwrap();
        assert_eq!(created.into_inner().verdict(), pb::Verdict::Created);

        let sending = Arc::clone(&telemetry);
        assert!(tokio::task::spawn_blocking(move || sending.flush()).await.unwrap());
        let spans = collector.exported_spans(6, Duration::from_secs(10)).await;
        let metrics = collector.exported_metrics(1, Duration::from_secs(10)).await;

        stops.send(Stop::Now).await.unwrap();
        drop(client);
        tokio::time::timeout(Duration::from_secs(10), server)
          .await
          .expect("the server did not stop")
          .unwrap()
          .unwrap();
        (spans, metrics)
      })
    });

    // Read as the collector's own rendering of what it got.
    let metrics = metrics
      .iter()
      .flat_map(|push| {
        serde_json::to_value(push).unwrap()["metrics"]
          .as_array()
          .unwrap()
          .clone()
      })
      .collect();
    let page = telemetry.page().unwrap();
    let stopping = Arc::clone(&telemetry);
    runtime.block_on(async move {
      tokio::task::spawn_blocking(move || stopping.shutdown()).await.unwrap();
    });
    Received { spans, metrics, page }
  }

  /// A request is one trace: the caller's, if it came with one. The call is a span in it, and inside the call
  /// are the things it waited for, each a span of its own, with what came of the call written on it.
  #[test]
  fn a_request_reaches_the_collector_as_one_trace() {
    let Received { spans, .. } = one_request_seen_by_a_collector();
    assert!(
      spans.iter().all(|span| span.trace_id == TRACE),
      "one trace, the caller's: {spans:#?}"
    );
    let named = |name: &str| -> Vec<&ExportedSpan> { spans.iter().filter(|span| span.name == name).collect() };
    let one = |name: &str| -> &ExportedSpan {
      let found = named(name);
      assert_eq!(found.len(), 1, "one `{name}` span: {spans:#?}");
      found[0]
    };
    let call = one("recordsimilarity.v1.RecordSimilarity/Evaluate");
    assert_eq!(call.parent_span_id, CALLER, "the call continues the caller's span");
    assert_eq!(call.kind, "SPAN_KIND_SERVER");
    let evaluate = one("evaluate");
    assert_eq!(evaluate.parent_span_id, call.span_id);
    // The collector shows an attribute's value as the protocol carries it, type and all.
    let says = |span: &ExportedSpan, key: &str, value: &str| {
      span.attributes.get(key).is_some_and(|carried| carried.contains(value))
    };
    for (key, value) in [
      ("record.id", "StringValue(\"p1\")"),
      ("verdict", "StringValue(\"Created\")"),
      ("mode", "StringValue(\"Enforce\")"),
      ("stored", "BoolValue(true)"),
      ("revision", "IntValue(1)"),
    ] {
      assert!(says(evaluate, key, value), "{key} is {value}: {evaluate:#?}");
    }

    // What the call waited for, each inside it.
    let lock = one("lock");
    let embed = one("embed");
    let stores = named("store");
    for waited in [lock, embed].into_iter().chain(stores.iter().copied()) {
      assert_eq!(waited.parent_span_id, evaluate.span_id, "{waited:#?}");
      assert!(waited.start_time_unix_nano >= evaluate.start_time_unix_nano);
      assert!(waited.end_time_unix_nano <= evaluate.end_time_unix_nano);
    }
    assert!(says(embed, "pass_texts", "IntValue(1)"), "{embed:#?}");
    assert!(embed.attributes.contains_key("waited_ms"), "{embed:#?}");
    assert_eq!(stores.len(), 2, "{stores:#?}");
    for operation in ["get", "put_if"] {
      let wanted = format!("StringValue(\"{operation}\")");
      let found = stores.iter().filter(|store| says(store, "operation", &wanted)).count();
      assert_eq!(found, 1, "the read, and the write on its own task: {stores:#?}");
    }
    assert_eq!(spans.len(), 6, "these, and nothing for the health check: {spans:#?}");
  }

  /// The few metrics arrive at the same collector, under the names `metrics.rs` gives them, and the page shows
  /// the same numbers.
  #[test]
  fn the_metrics_reach_the_same_collector_and_the_page_agrees() {
    let Received {
      metrics: sent, page, ..
    } = one_request_seen_by_a_collector();
    let metric = |name: &str| sent.iter().find(|metric| metric["name"] == name).unwrap();
    let verdicts = &metric(crate::metrics::EVALUATIONS)["data"]["Sum"];
    assert_eq!(verdicts["is_monotonic"], true, "{verdicts}");
    let carries = |point: &serde_json::Value, key: &str, value: &str| {
      point["attributes"][key]
        .as_str()
        .is_some_and(|carried| carried.contains(&format!("StringValue(\"{value}\")")))
    };
    let created = verdicts["data_points"]
      .as_array()
      .unwrap()
      .iter()
      .find(|point| carries(point, "verdict", "created") && carries(point, "mode", "enforce"))
      .unwrap();
    assert_eq!(created["value"]["AsInt"], 1, "{verdicts}");
    assert_eq!(
      verdicts["data_points"].as_array().unwrap().len(),
      12,
      "the rest are there, at zero"
    );
    let pass = metric(crate::metrics::PASS_SECONDS);
    assert_eq!(pass["unit"], "s");
    let passes = &pass["data"]["Histogram"]["data_points"][0];
    assert_eq!(passes["count"], 1, "one forward pass: {pass}");
    assert_eq!(passes["explicit_bounds"].as_array().unwrap().len(), 11);
    for name in [
      crate::metrics::WAITING,
      crate::metrics::REFUSED,
      crate::metrics::ABANDONED,
      crate::metrics::UNANSWERED_WRITES,
    ] {
      assert!(sent.iter().any(|metric| metric["name"] == name), "{name} was sent");
    }
    assert_eq!(sent.len(), 6, "these and nothing else: {sent:#?}");
    let created = "record_similarity_evaluations_total{mode=\"enforce\",verdict=\"created\"}";
    assert!(
      page.lines().any(|line| line == format!("{created} 1")),
      "and the page says the same: {page}"
    );
  }

  /// A collector that is not there costs the server nothing but the sending: requests are served, the metrics
  /// are kept, and asking to send says that it failed.
  #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
  async fn a_collector_that_is_not_there_does_not_stop_anything() {
    // A port nothing is listening on: bound, read, and let go.
    let nowhere = TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap();
    let settings = Settings {
      otlp_endpoint: Some(format!("http://{nowhere}")),
      metrics_every: Duration::from_hours(1),
      metrics_page: true,
      trace_sample_ratio: 1.0,
    };
    let (telemetry, tracing) = Telemetry::start(&settings).unwrap();
    assert!(tracing.is_some());
    let telemetry = Arc::new(telemetry);
    telemetry.metrics().abandoned();
    let sending = Arc::clone(&telemetry);
    assert!(!tokio::task::spawn_blocking(move || sending.flush()).await.unwrap());
    assert_eq!(
      testing::counted(&telemetry, crate::metrics::ABANDONED),
      Some(1.0),
      "counted all the same"
    );
  }
}
