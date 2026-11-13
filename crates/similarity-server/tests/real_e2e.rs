//! The whole stack, for real: the shipped configuration, the real model, a real Qdrant, and a gRPC client.
//!
//! Everything else in the workspace tests one layer against a stand-in for the next. This is the one test where
//! nothing is faked, so it is the one that proves the pieces fit and that the shipped thresholds do what the README
//! says on realistic products. Ignored by default because it needs the model and a Qdrant server:
//!
//! ```text
//! just fetch-model && just qdrant-up && just test-real
//! ```

#![allow(
  clippy::unwrap_used,
  clippy::expect_used,
  clippy::panic,
  clippy::indexing_slicing,
  missing_docs
)]
#![allow(clippy::arithmetic_side_effects, clippy::float_cmp, clippy::too_many_lines)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use similarity_core::{Device, Embedder};
use similarity_proto::v1 as pb;
use similarity_proto::v1::record_similarity_client::RecordSimilarityClient;
use similarity_server::anchors;
use similarity_server::audit::{self, Kind};
use similarity_server::config::Config;
use similarity_server::engine::{Engine, EvaluateInput, RecordInput, Restored, WriteMode, store_layout};
use similarity_server::import::{self, Options, Summary};
use similarity_server::worker::{DirectEmbedder, EmbedWorker};
use similarity_store::{QdrantStore, VectorStore};
use tokio::sync::oneshot;
use tonic::transport::Channel;

fn record(id: &str, brand_id: &str, fields: &[(&str, &str)]) -> pb::Record {
  pb::Record {
    id: Some(id.to_owned()),
    brand_id: Some(brand_id.to_owned()),
    fields: fields
      .iter()
      .map(|(name, text)| ((*name).to_owned(), (*text).to_owned()))
      .collect::<HashMap<_, _>>(),
    quantities: HashMap::new(),
  }
}

/// `record` with quantities, each a single value: `(name, value, unit)`.
fn measured(id: &str, brand_id: &str, fields: &[(&str, &str)], quantities: &[(&str, f64, &str)]) -> pb::Record {
  pb::Record {
    quantities: quantities
      .iter()
      .map(|(name, value, unit)| {
        let quantity = pb::Quantity {
          min: *value,
          max: *value,
          unit: (*unit).to_owned(),
        };
        ((*name).to_owned(), quantity)
      })
      .collect(),
    ..record(id, brand_id, fields)
  }
}

fn evaluate(record: pb::Record) -> pb::EvaluateRequest {
  pb::EvaluateRequest {
    record: Some(record),
    ..pb::EvaluateRequest::default()
  }
}

const GUMMIES: [(&str, &str); 4] = [
  ("name", "Blueberry Gummies 100mg"),
  ("category", "Edibles > Gummies"),
  ("brand", "WYLD"),
  (
    "description",
    "Ten vegan gummies made with real fruit and infused with indica-enhanced botanical terpenes. 10mg THC per \
     piece, 100mg per package.",
  ),
];
const GUMMIES_EDITED: [(&str, &str); 4] = [
  ("name", "Blueberry Gummies 100mg (Indica)"),
  GUMMIES[1],
  GUMMIES[2],
  (
    "description",
    "Ten vegan gummies made with real fruit. Indica-enhanced with botanical terpenes. 10mg THC each, 100mg per \
     pack. Lab tested.",
  ),
];
const VAPE: [(&str, &str); 4] = [
  ("name", "Sour Diesel Vape Cartridge 1g"),
  ("category", "Vape Pens > Cartridges"),
  ("brand", "STIIIZY"),
  (
    "description",
    "One gram 510-thread cartridge filled with premium distillate and cannabis-derived terpenes. Energizing sativa.",
  ),
];
const RASPBERRY_GUMMIES: [(&str, &str); 4] = [
  ("name", "Raspberry Gummies 100mg"),
  GUMMIES[1],
  GUMMIES[2],
  (
    "description",
    "Ten vegan raspberry gummies with sativa-enhanced terpenes. 10mg THC per piece.",
  ),
];

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs ./models/mdbr-leaf-mt and a Qdrant server at $QDRANT_URL; run with `just test-real`"]
async fn the_shipped_configuration_catches_a_recycled_product_end_to_end() {
  let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
  let mut config = Config::parse(include_str!("../../../config.example.toml")).unwrap();
  config.model.dir = root.join(&config.model.dir);
  config.qdrant.url = std::env::var("QDRANT_URL").unwrap_or(config.qdrant.url);
  config.qdrant.collection = format!(
    "e2e_{}_{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .unwrap()
      .as_nanos()
  );

  let embedder = Embedder::load(&config.model.dir, Device::flex(), &config.model.embedder_options()).unwrap();
  let worker = EmbedWorker::spawn(embedder, config.model.queue_depth, config.model.batch_size).unwrap();
  let schema = config.schema().unwrap();
  let (layout, dimension) = store_layout(&schema, &worker);
  let store = QdrantStore::connect(
    &config.qdrant.settings(std::env::var("QDRANT_API_KEY").ok()),
    layout,
    dimension,
  )
  .await
  .unwrap();
  let engine = Arc::new(Engine::new(store.clone(), worker, schema, config.policy().unwrap()));

  let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
  let address = listener.local_addr().unwrap();
  let (stop, stopped) = oneshot::channel::<()>();
  let server = tokio::spawn(similarity_server::serve(engine, listener, async {
    let _ = stopped.await;
  }));
  let channel = Channel::from_shared(format!("http://{address}"))
    .unwrap()
    .connect()
    .await
    .unwrap();

  let outcome = tokio::spawn(scenario(RecordSimilarityClient::new(channel))).await;

  // Clean up before reporting, so a failed assertion does not leave a collection behind.
  stop.send(()).unwrap();
  server.await.unwrap().unwrap();
  store.delete_collection().await.unwrap();
  outcome.unwrap();
}

async fn scenario(mut client: RecordSimilarityClient<Channel>) {
  let description = client.describe(pb::DescribeRequest {}).await.unwrap().into_inner();
  assert_eq!(
    (description.dimension, description.composite_dimension),
    (256, 1024),
    "four text fields; the quantities take no room in a vector"
  );
  let quantities: Vec<&str> = description
    .quantities
    .iter()
    .map(|quantity| quantity.name.as_str())
    .collect();
  assert_eq!(
    quantities,
    ["cbd_per_item", "thc_per_item", "volume_per_item", "weight_per_item"]
  );
  assert!(
    description.model_fingerprint.ends_with("-d256-t256"),
    "{}",
    description.model_fingerprint
  );

  // A new product becomes the baseline.
  let created = client
    .evaluate(evaluate(record("sku-1", "brand-1", &GUMMIES)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(
    (created.verdict(), created.stored, created.revision),
    (pb::Verdict::Created, true, 1)
  );

  // An ordinary edit (name suffix, reworded description) is accepted and stored.
  let edited = client
    .evaluate(evaluate(record("sku-1", "brand-1", &GUMMIES_EDITED)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(edited.verdict(), pb::Verdict::Accepted, "{edited:?}");
  assert!(
    edited.stored && edited.similarity.unwrap() > 0.93,
    "{:?}",
    edited.similarity
  );
  assert_eq!(edited.revision, 2);
  let unchanged = edited
    .fields
    .iter()
    .filter(|field| field.change() == pb::FieldChange::Unchanged)
    .count();
  assert_eq!(unchanged, 2, "category and brand were not touched");

  // Turning the gummies into a vape cartridge is rejected, with reasons, and nothing is written.
  let recycled = client
    .evaluate(evaluate(record("sku-1", "brand-1", &VAPE)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(recycled.verdict(), pb::Verdict::Rejected, "{recycled:?}");
  assert!(
    !recycled.stored && recycled.similarity.unwrap() < 0.75,
    "{:?}",
    recycled.similarity
  );
  assert_eq!(recycled.reasons.len(), recycled.reason_details.len());
  assert!(
    recycled
      .reason_details
      .iter()
      .any(|reason| reason.code() == pb::ReasonCode::BelowThreshold),
    "{:?}",
    recycled.reasons
  );
  assert!(
    recycled
      .reason_details
      .iter()
      .any(|reason| reason.code() == pb::ReasonCode::FieldBelowFloor && reason.subject.as_deref() == Some("name")),
    "{:?}",
    recycled.reasons
  );
  assert_eq!(recycled.revision, 2);

  // The subtler recycle: only the name is swapped for another product's, everything else is left to prop it up.
  let renamed = [
    ("name", "OG Kush Pre-Roll 1g"),
    GUMMIES_EDITED[1],
    GUMMIES_EDITED[2],
    GUMMIES_EDITED[3],
  ];
  let sneaky = client
    .evaluate(evaluate(record("sku-1", "brand-1", &renamed)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(sneaky.verdict(), pb::Verdict::Rejected, "{sneaky:?}");
  assert!(
    sneaky
      .fields
      .iter()
      .any(|field| field.field == "name" && field.below_floor)
  );

  // The stored baseline survived both attempts: re-submitting the accepted edit is a no-op at similarity 1.
  let again = client
    .evaluate(evaluate(record("sku-1", "brand-1", &GUMMIES_EDITED)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(
    (again.verdict(), again.stored, again.revision),
    (pb::Verdict::Accepted, false, 2)
  );
  assert!((again.similarity.unwrap() - 1.0).abs() < 1e-6);

  // Neighbor search ranks a sibling product above an unrelated one, on the same scale as the verdicts.
  client
    .evaluate(evaluate(record("sku-2", "brand-1", &RASPBERRY_GUMMIES)))
    .await
    .unwrap();
  client
    .evaluate(evaluate(record("sku-3", "brand-2", &VAPE)))
    .await
    .unwrap();
  let around = |brand_id: Option<&str>| pb::FindSimilarRequest {
    query: Some(pb::find_similar_request::Query::Id("sku-1".to_owned())),
    brand_id: brand_id.map(str::to_owned),
    ..pb::FindSimilarRequest::default()
  };
  let neighbors = client.find_similar(around(None)).await.unwrap().into_inner().neighbors;
  let ids: Vec<&str> = neighbors.iter().map(|neighbor| neighbor.id.as_str()).collect();
  assert_eq!(ids, ["sku-2", "sku-3"]);
  assert!(
    neighbors[0].similarity > 0.8 && neighbors[1].similarity < 0.75,
    "{neighbors:?}"
  );
  assert!(
    (neighbors[1].similarity - recycled.similarity.unwrap()).abs() < 0.02,
    "search and evaluate agree"
  );

  let own = client
    .find_similar(around(Some("brand-1")))
    .await
    .unwrap()
    .into_inner()
    .neighbors;
  assert_eq!(
    own.iter().map(|neighbor| neighbor.id.as_str()).collect::<Vec<_>>(),
    ["sku-2"]
  );

  // "What is this most like?" for a record that is not stored.
  let probe = pb::FindSimilarRequest {
    query: Some(pb::find_similar_request::Query::Record(pb::Record {
      id: None,
      brand_id: None,
      fields: HashMap::from([("name".to_owned(), "sour diesel 510 cart".to_owned())]),
      quantities: HashMap::new(),
    })),
    limit: 1,
    ..pb::FindSimilarRequest::default()
  };
  assert_eq!(
    client.find_similar(probe).await.unwrap().into_inner().neighbors[0].id,
    "sku-3"
  );

  // "Which stored product is this menu item for?" The menu item has a name and nothing else, in a
  // retailer's words. The vape it describes has every field filled in, so the search scores it low; scored on
  // the name alone, which is all the two share, it is well ahead. The shipped configuration has no match
  // threshold, so the candidates are ranked and nothing is claimed.
  let menu_item = pb::MatchRequest {
    menu_item: Some(pb::Record {
      id: None,
      brand_id: None,
      fields: HashMap::from([("name".to_owned(), "sour diesel 510 cart".to_owned())]),
      quantities: HashMap::new(),
    }),
    ..pb::MatchRequest::default()
  };
  let matched = client.r#match(menu_item.clone()).await.unwrap().into_inner();
  assert_eq!(matched.decision(), pb::MatchDecision::NotJudged);
  assert_eq!(matched.threshold, None);
  assert_eq!(matched.candidates.len(), 3, "{matched:?}");
  let best = &matched.candidates[0];
  assert_eq!((best.id.as_str(), best.brand_id.as_deref()), ("sku-3", Some("brand-2")));
  assert!(
    best.similarity > best.search_similarity + 0.1,
    "the match score is not the search score: {best:?}"
  );
  assert!(best.similarity > matched.candidates[1].similarity + 0.05, "{matched:?}");
  let scored: Vec<&str> = best
    .fields
    .iter()
    .filter(|field| field.similarity.is_some())
    .map(|field| field.field.as_str())
    .collect();
  assert_eq!(scored, ["name"], "only the field both sides have");
  assert!(
    best
      .fields
      .iter()
      .all(|field| field.field == "name" || field.change() == pb::FieldChange::Removed)
  );
  // Within the other brand's records there is no vape to find, and a threshold says so.
  let elsewhere = pb::MatchRequest {
    brand_id: Some("brand-1".to_owned()),
    threshold: Some(best.similarity),
    ..menu_item
  };
  let elsewhere = client.r#match(elsewhere).await.unwrap().into_inner();
  assert_eq!(elsewhere.decision(), pb::MatchDecision::NoMatch, "{elsewhere:?}");
  assert!(elsewhere.candidates.iter().all(|candidate| candidate.id != "sku-3"));

  // A reviewer can force the change through; after that the vape is the record.
  let force = pb::EvaluateRequest {
    mode: pb::WriteMode::Force.into(),
    ..evaluate(record("sku-1", "brand-1", &VAPE))
  };
  let forced = client.evaluate(force).await.unwrap().into_inner();
  assert_eq!(
    (forced.verdict(), forced.stored, forced.revision),
    (pb::Verdict::Rejected, true, 3)
  );
  let settled = client
    .evaluate(evaluate(record("sku-1", "brand-1", &VAPE)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(settled.verdict(), pb::Verdict::Accepted);
  assert!(
    (settled.anchor_similarity.unwrap() - 1.0).abs() < 1e-4,
    "the forced write reset the anchor"
  );

  // Quantities ride along through Qdrant. The shipped configuration sets no limits, so they are reported and
  // never reject: the same vape at half the size, with its potency rewritten in other units, is still accepted.
  let sized = [("weight_per_item", 1.0, "g"), ("thc_per_item", 850.0, "mg")];
  let first = client
    .evaluate(evaluate(measured("sku-1", "brand-1", &VAPE, &sized)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(first.verdict(), pb::Verdict::Accepted);
  assert!(first.stored, "gaining quantities is a change worth writing");
  assert!(
    first
      .quantities
      .iter()
      .all(|quantity| quantity.change() == pb::QuantityChange::Added),
    "{:?}",
    first.quantities
  );
  assert!(!first.measure_changed);

  let resized = [("weight_per_item", 500.0, "mg"), ("thc_per_item", 0.85, "g")];
  let halved = client
    .evaluate(evaluate(measured("sku-1", "brand-1", &VAPE, &resized)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(halved.verdict(), pb::Verdict::Accepted, "{halved:?}");
  assert!((halved.similarity.unwrap() - 1.0).abs() < 1e-6, "no text changed");
  assert!(halved.stored, "a quantity changed, so the record did");
  let quantity = |name: &str| halved.quantities.iter().find(|score| score.field == name).unwrap();
  let weight = quantity("weight_per_item");
  assert_eq!(weight.change(), pb::QuantityChange::Modified);
  assert!((weight.factor.unwrap() - 2.0).abs() < 1e-9, "{weight:?}");
  assert_eq!(
    (weight.previous_unit.as_deref(), weight.unit.as_deref()),
    (Some("g"), Some("mg"))
  );
  assert!(!weight.beyond_limit, "no limit is shipped");
  let thc = quantity("thc_per_item");
  assert_eq!(thc.change(), pb::QuantityChange::Unchanged, "850 mg is 0.85 g");
  assert_eq!(thc.factor, Some(1.0));

  // What Qdrant holds is what was sent: sending it again changes nothing.
  let again = client
    .evaluate(evaluate(measured("sku-1", "brand-1", &VAPE, &resized)))
    .await
    .unwrap()
    .into_inner();
  assert!(!again.stored);
  assert_eq!(again.revision, halved.revision);

  // Deleting starts over.
  assert!(
    client
      .delete(pb::DeleteRequest {
        id: "sku-1".to_owned(),
        brand_id: Some("brand-1".to_owned()),
      })
      .await
      .unwrap()
      .into_inner()
      .deleted
  );
  let reborn = client
    .evaluate(evaluate(record("sku-1", "brand-1", &GUMMIES)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(reborn.verdict(), pb::Verdict::Created);
}

/// One line of an export: the gummies as they are now, and optionally as they were created.
fn export_line(id: &str, name: &str, original_name: Option<&str>) -> String {
  let record = |name: &str, weight: f64| {
    serde_json::json!({
      "name": name,
      "category": GUMMIES[1].1,
      "brand": GUMMIES[2].1,
      "description": GUMMIES[3].1,
      "weight_per_item": {"value": weight, "unit": "g"},
    })
  };
  let mut line = serde_json::json!({"id": id, "brand_id": "brand-1", "record": record(name, 40.0)});
  if let Some(original) = original_name {
    line["anchor"] = record(original, 20.0);
  }
  line.to_string()
}

/// An engine on the real model and a new Qdrant collection of its own.
struct Importing {
  engine: Engine<QdrantStore, DirectEmbedder>,
  /// The store the engine writes to.
  store: QdrantStore,
  /// What both were made from.
  config: Config,
  /// The layout and width the collection was connected to with.
  layout: (String, usize),
}

async fn importing(prefix: &str) -> Importing {
  let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
  let mut config = Config::parse(include_str!("../../../config.example.toml")).unwrap();
  config.model.dir = root.join(&config.model.dir);
  config.qdrant.url = std::env::var("QDRANT_URL").unwrap_or(config.qdrant.url);
  config.qdrant.collection = format!(
    "{prefix}_{}_{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .unwrap()
      .as_nanos()
  );
  let embedder =
    DirectEmbedder(Embedder::load(&config.model.dir, Device::flex(), &config.model.embedder_options()).unwrap());
  let schema = config.schema().unwrap();
  let (layout, dimension) = store_layout(&schema, &embedder);
  let store = QdrantStore::connect(
    &config.qdrant.settings(std::env::var("QDRANT_API_KEY").ok()),
    layout.as_str(),
    dimension,
  )
  .await
  .unwrap();
  let engine = Engine::new(store.clone(), embedder, schema, config.policy().unwrap());
  Importing {
    engine,
    store,
    config,
    layout: (layout, dimension),
  }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs ./models/mdbr-leaf-mt and a Qdrant server at $QDRANT_URL; run with `just test-real`"]
async fn an_imported_catalog_audits_clean_in_qdrant_and_what_is_done_to_it_afterwards_is_found() {
  let Importing {
    engine,
    store,
    config,
    layout: (layout, dimension),
  } = importing("audit").await;
  let schema = config.schema().unwrap();
  let export = [
    export_line("sku-1", "Blueberry Gummies 100mg", None),
    export_line("sku-2", "Blueberry Gummies 100mg (Indica)", Some("Blueberry Gummies")),
    export_line("sku-3", "Raspberry Gummies 100mg", None),
    export_line("sku-4", "Peach Gummies 100mg", None),
    export_line("sku-5", "Mango Gummies 100mg", None),
  ]
  .join("\n");

  let settings = config.qdrant.settings(std::env::var("QDRANT_API_KEY").ok());
  let outcome = tokio::spawn(async move {
    let options = Options {
      batch_records: 2,
      stop_on_error: false,
    };
    let imported = import::run(&engine, export.as_bytes(), options, |_| {}, |_| {})
      .await
      .unwrap();
    assert_eq!(imported.imported, 5);

    // The audit looks at the collection the way the command does: opened, under the layout the engine writes,
    // and read in pages smaller than the collection.
    let opened = QdrantStore::open(&settings, layout, dimension).await.unwrap();
    let audit = || async {
      let mut findings = Vec::new();
      let options = audit::Options {
        page_records: 2,
        exported_at: None,
      };
      let summary = audit::run(
        &opened,
        &schema,
        export.as_bytes(),
        options,
        |finding| findings.push(finding.clone()),
        |_| {},
      )
      .await
      .unwrap();
      (summary, findings)
    };
    let (clean, findings) = audit().await;
    assert_eq!(findings, []);
    assert_eq!(
      (clean.export_records, clean.points, clean.agree, clean.differences()),
      (5, 5, 5, 0)
    );

    // An edit the service accepted and the caller's database never got, a record the service forgot, and one it
    // was given that the catalog does not have.
    let edit = EvaluateInput {
      record: RecordInput {
        id: Some("sku-1".to_owned()),
        brand_id: Some("brand-1".to_owned()),
        fields: vec![
          ("name".to_owned(), "Blueberry Gummies 100mg - Indica".to_owned()),
          ("category".to_owned(), GUMMIES[1].1.to_owned()),
          ("brand".to_owned(), GUMMIES[2].1.to_owned()),
          ("description".to_owned(), GUMMIES[3].1.to_owned()),
        ],
        quantities: vec![(
          "weight_per_item".to_owned(),
          similarity_core::Quantity::single(0.04, "kg"),
        )],
      },
      mode: WriteMode::Enforce,
      threshold: None,
    };
    assert!(engine.evaluate(edit).await.unwrap().stored);
    engine
      .delete("sku-3".to_owned(), Some("brand-1".to_owned()))
      .await
      .unwrap();
    let extra = export_line("sku-9", "Sour Diesel Live Resin Cartridge 1g", None);
    import::run(&engine, extra.as_bytes(), options, |_| {}, |_| {})
      .await
      .unwrap();

    let (summary, findings) = audit().await;
    let mut found: Vec<(Kind, Option<&str>, Option<&str>)> = findings
      .iter()
      .map(|finding| (finding.kind, finding.id.as_deref(), finding.field.as_deref()))
      .collect();
    found.sort();
    assert_eq!(
      found,
      [
        (Kind::NotStored, Some("sku-3"), None),
        (Kind::NotInExport, Some("sku-9"), None),
        // The name, and nothing else of that record: its weight is the same amount in another unit.
        (Kind::TextDiffers, Some("sku-1"), Some("name")),
      ]
    );
    assert_eq!(
      (summary.points, summary.agree, summary.differ, summary.differences()),
      (5, 3, 1, 3)
    );
  })
  .await;

  store.delete_collection().await.unwrap();
  outcome.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs ./models/mdbr-leaf-mt and a Qdrant server at $QDRANT_URL; run with `just test-real`"]
async fn anchors_backed_up_from_qdrant_are_restored_to_a_catalog_that_was_lost_and_imported_again() {
  let lines = |with_originals: bool| {
    [
      export_line("sku-1", "Blueberry Gummies 100mg", None),
      export_line(
        "sku-2",
        "Blueberry Gummies 100mg (Indica)",
        with_originals.then_some("Blueberry Gummies"),
      ),
      export_line("sku-3", "Raspberry Gummies 100mg", None),
    ]
    .join("\n")
  };
  let options = Options {
    batch_records: 2,
    stop_on_error: false,
  };

  // A collection whose second record has come some way from what it was created as, and a backup of it, taken
  // from the collection opened the way the command opens it.
  let before = importing("anchors_before").await;
  let (backup, had) = {
    let Importing {
      engine,
      store,
      config,
      layout: (layout, dimension),
    } = before;
    let settings = config.qdrant.settings(std::env::var("QDRANT_API_KEY").ok());
    let outcome = tokio::spawn(async move {
      import::run(&engine, lines(true).as_bytes(), options, |_| {}, |_| {})
        .await
        .unwrap();
      let opened = QdrantStore::open(&settings, layout.as_str(), dimension).await.unwrap();
      let header = anchors::file::Header {
        format: anchors::file::FORMAT,
        fingerprint: "the pinned model".to_owned(),
        layout,
        dimension,
        collection: settings.collection.clone(),
        taken_at: 1_794_591_798,
      };
      let (bytes, exported) = anchors::export(&opened, &header, Vec::new(), 2, |_, _| {}, |_| {})
        .await
        .unwrap();
      assert_eq!(
        (exported.records, exported.other_layout, exported.unreadable),
        (3, 0, 0)
      );
      (bytes, opened.get("sku-2").await.unwrap().unwrap())
    })
    .await;
    // The collection is lost, here on purpose.
    store.delete_collection().await.unwrap();
    outcome.unwrap()
  };
  assert!(
    similarity_core::vector::cosine(&had.anchor, &had.vectors.composite).unwrap() < 0.999,
    "the record had come some way from its anchor"
  );

  // A new collection, loaded from a database that knows what each product is and not what it was.
  let Importing {
    engine,
    store,
    layout: (layout, dimension),
    ..
  } = importing("anchors_after").await;
  let reader = store.clone();
  let outcome = tokio::spawn(async move {
    import::run(&engine, lines(false).as_bytes(), options, |_| {}, |_| {})
      .await
      .unwrap();
    let imported = reader.get("sku-2").await.unwrap().unwrap();
    assert!(similarity_core::vector::cosine(&imported.anchor, &imported.vectors.composite).unwrap() > 0.9999);

    let restore = || async {
      let mut told = Vec::new();
      let summary = anchors::restore(
        &engine,
        (&layout, dimension),
        || Ok(backup.as_slice()),
        true,
        |id, outcome| told.push((id.to_owned(), *outcome.as_ref().unwrap())),
        |_| {},
      )
      .await
      .unwrap();
      told.sort();
      (summary, told)
    };
    let (summary, told) = restore().await;
    assert_eq!(
      told,
      [
        ("sku-1".to_owned(), Restored::Already),
        ("sku-2".to_owned(), Restored::Restored),
        ("sku-3".to_owned(), Restored::Already),
      ],
      "an anchor that went into Qdrant and came back is still told from one that did not move"
    );
    assert_eq!((summary.records, summary.failed), (3, 0));

    // The anchor is the one the lost collection held, through a file and two round trips to Qdrant.
    let restored = reader.get("sku-2").await.unwrap().unwrap();
    assert!(similarity_core::vector::cosine(&restored.anchor, &had.anchor).unwrap() > 0.999_999);
    assert_eq!(restored.anchor_quantities, had.anchor_quantities);
    assert_eq!((restored.revision, &restored.vectors.hashes), (2, &had.vectors.hashes));
    let (_, again) = restore().await;
    assert!(
      again.iter().all(|(_, outcome)| *outcome == Restored::Already),
      "{again:?}"
    );
  })
  .await;
  store.delete_collection().await.unwrap();
  outcome.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs ./models/mdbr-leaf-mt and a Qdrant server at $QDRANT_URL; run with `just test-real`"]
async fn a_catalog_is_imported_into_qdrant_with_its_original_versions_and_a_second_run_resumes() {
  let Importing { engine, store, .. } = importing("import").await;

  let export = [
    export_line("sku-1", "Blueberry Gummies 100mg", None),
    // This one has been edited since it was created, and the export still knows what it was.
    export_line("sku-2", "Blueberry Gummies 100mg (Indica)", Some("Blueberry Gummies")),
    export_line("sku-3", "Raspberry Gummies 100mg", None),
    r#"{"id": "bad", "record": {"category": "no name"}}"#.to_owned(),
  ]
  .join("\n");
  let options = Options {
    batch_records: 2,
    stop_on_error: false,
  };
  let reader = store.clone();
  let outcome = tokio::spawn(async move {
    let run = || async {
      let mut failures = Vec::new();
      let summary = import::run(
        &engine,
        export.as_bytes(),
        options,
        |failure| failures.push(failure.line),
        |_| {},
      )
      .await
      .unwrap();
      (summary, failures)
    };
    let (first, failures) = run().await;
    assert_eq!(
      first,
      Summary {
        records: 4,
        imported: 3,
        skipped: 0,
        failed: 1
      }
    );
    assert_eq!(failures, [4]);

    // What Qdrant holds: a plain record is its own anchor, and one with an original version is not.
    let plain = reader.get("sku-1").await.unwrap().unwrap();
    assert_eq!((plain.revision, plain.brand_id.as_deref()), (1, Some("brand-1")));
    assert!(similarity_core::vector::cosine(&plain.anchor, &plain.vectors.composite).unwrap() > 0.9999);
    let edited = reader.get("sku-2").await.unwrap().unwrap();
    let agreement = similarity_core::vector::cosine(&edited.anchor, &edited.vectors.composite).unwrap();
    assert!(agreement < 0.999, "the anchor is the original version: {agreement}");
    assert_eq!(edited.quantities["weight_per_item"].max, 40.0);
    assert_eq!(
      edited.anchor_quantities["weight_per_item"].max, 20.0,
      "and so is its weight"
    );

    // Through the engine, the record reports the drift it had already accumulated before it was imported.
    let preview = EvaluateInput {
      record: RecordInput {
        id: Some("sku-2".to_owned()),
        brand_id: Some("brand-1".to_owned()),
        fields: vec![
          ("name".to_owned(), "Blueberry Gummies 100mg (Indica)".to_owned()),
          ("category".to_owned(), GUMMIES[1].1.to_owned()),
          ("brand".to_owned(), GUMMIES[2].1.to_owned()),
          ("description".to_owned(), GUMMIES[3].1.to_owned()),
        ],
        quantities: Vec::new(),
      },
      mode: WriteMode::DryRun,
      threshold: None,
    };
    let output = engine.evaluate(preview).await.unwrap();
    assert!(
      (output.similarity.unwrap() - 1.0).abs() < 1e-5,
      "nothing changed since the import"
    );
    assert!(
      output.anchor_similarity.unwrap() < 0.999,
      "{:?}",
      output.anchor_similarity
    );

    // Running it again is how an interrupted import is resumed: everything stored is skipped.
    let (second, _) = run().await;
    assert_eq!(
      second,
      Summary {
        records: 4,
        imported: 0,
        skipped: 3,
        failed: 1
      }
    );
    assert_eq!(reader.get("sku-2").await.unwrap().unwrap(), edited);
  })
  .await;

  // Clean up before reporting, so a failed assertion does not leave a collection behind.
  store.delete_collection().await.unwrap();
  outcome.unwrap();
}
