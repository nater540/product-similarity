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
use similarity_server::config::Config;
use similarity_server::engine::{Engine, store_layout};
use similarity_server::worker::EmbedWorker;
use similarity_store::QdrantStore;
use tokio::sync::oneshot;
use tonic::transport::Channel;

fn record(id: &str, tenant: &str, fields: &[(&str, &str)]) -> pb::Record {
  pb::Record {
    id: Some(id.to_owned()),
    tenant: Some(tenant.to_owned()),
    fields: fields
      .iter()
      .map(|(name, text)| ((*name).to_owned(), (*text).to_owned()))
      .collect::<HashMap<_, _>>(),
    quantities: HashMap::new(),
  }
}

/// `record` with quantities, each a single value: `(name, value, unit)`.
fn measured(id: &str, tenant: &str, fields: &[(&str, &str)], quantities: &[(&str, f64, &str)]) -> pb::Record {
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
    ..record(id, tenant, fields)
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
    .evaluate(evaluate(record("sku-1", "merchant-1", &GUMMIES)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(
    (created.verdict(), created.stored, created.revision),
    (pb::Verdict::Created, true, 1)
  );

  // An ordinary edit (name suffix, reworded description) is accepted and stored.
  let edited = client
    .evaluate(evaluate(record("sku-1", "merchant-1", &GUMMIES_EDITED)))
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
    .evaluate(evaluate(record("sku-1", "merchant-1", &VAPE)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(recycled.verdict(), pb::Verdict::Rejected, "{recycled:?}");
  assert!(
    !recycled.stored && recycled.similarity.unwrap() < 0.75,
    "{:?}",
    recycled.similarity
  );
  assert!(
    recycled
      .reasons
      .iter()
      .any(|reason| reason.contains("below the threshold")),
    "{:?}",
    recycled.reasons
  );
  assert!(
    recycled.reasons.iter().any(|reason| reason.contains("`name`")),
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
    .evaluate(evaluate(record("sku-1", "merchant-1", &renamed)))
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
    .evaluate(evaluate(record("sku-1", "merchant-1", &GUMMIES_EDITED)))
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
    .evaluate(evaluate(record("sku-2", "merchant-1", &RASPBERRY_GUMMIES)))
    .await
    .unwrap();
  client
    .evaluate(evaluate(record("sku-3", "merchant-2", &VAPE)))
    .await
    .unwrap();
  let around = |tenant: Option<&str>| pb::FindSimilarRequest {
    query: Some(pb::find_similar_request::Query::Id("sku-1".to_owned())),
    tenant: tenant.map(str::to_owned),
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
    .find_similar(around(Some("merchant-1")))
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
      tenant: None,
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

  // A reviewer can force the change through; after that the vape is the record.
  let force = pb::EvaluateRequest {
    mode: pb::WriteMode::Force.into(),
    ..evaluate(record("sku-1", "merchant-1", &VAPE))
  };
  let forced = client.evaluate(force).await.unwrap().into_inner();
  assert_eq!(
    (forced.verdict(), forced.stored, forced.revision),
    (pb::Verdict::Rejected, true, 3)
  );
  let settled = client
    .evaluate(evaluate(record("sku-1", "merchant-1", &VAPE)))
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
    .evaluate(evaluate(measured("sku-1", "merchant-1", &VAPE, &sized)))
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
    .evaluate(evaluate(measured("sku-1", "merchant-1", &VAPE, &resized)))
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
    .evaluate(evaluate(measured("sku-1", "merchant-1", &VAPE, &resized)))
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
        tenant: Some("merchant-1".to_owned()),
      })
      .await
      .unwrap()
      .into_inner()
      .deleted
  );
  let reborn = client
    .evaluate(evaluate(record("sku-1", "merchant-1", &GUMMIES)))
    .await
    .unwrap()
    .into_inner();
  assert_eq!(reborn.verdict(), pb::Verdict::Created);
}
