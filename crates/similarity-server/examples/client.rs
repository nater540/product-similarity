//! A minimal gRPC client, for poking a running server by hand and as a starting point for a real one.
//!
//! ```text
//! cargo run -p similarity-server --example client -- describe
//! cargo run -p similarity-server --example client -- evaluate sku-1 data/examples/gummies.json
//! cargo run -p similarity-server --example client -- evaluate sku-1 data/examples/vape.json --dry-run
//! cargo run -p similarity-server --example client -- similar sku-1
//! cargo run -p similarity-server --example client -- delete sku-1
//! ```
//!
//! A record file is one JSON object: a string value is a text field, an object value is a quantity, as
//! `{"value": 3.5, "unit": "g"}` or `{"min": 18, "max": 22, "unit": "%"}`.
//!
//! The server also exposes gRPC reflection, so `grpcurl -plaintext localhost:50051 list` works too.

#![allow(clippy::print_stdout, reason = "an example's whole job is to print")]

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};
use similarity_proto::v1 as pb;
use similarity_proto::v1::record_similarity_client::RecordSimilarityClient;
use similarity_server::offline::RecordFields;

#[derive(Debug, Parser)]
struct Cli {
  /// The server to talk to.
  #[arg(long, default_value = "http://127.0.0.1:50051")]
  server: String,
  #[command(subcommand)]
  command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
  /// Show the server's model, schema, and thresholds.
  Describe,
  /// Evaluate a record: a JSON object of field name to text, and quantity name to an object with a unit.
  Evaluate {
    /// The record's id.
    id: String,
    /// The JSON file holding the record's fields.
    fields: PathBuf,
    /// The record's tenant.
    #[arg(long)]
    tenant: Option<String>,
    /// Report the verdict without storing anything.
    #[arg(long, conflicts_with = "force")]
    dry_run: bool,
    /// Store the record whatever the verdict.
    #[arg(long)]
    force: bool,
  },
  /// List the stored records most similar to a stored record.
  Similar {
    /// The record's id.
    id: String,
    /// How many neighbors to list.
    #[arg(long, default_value_t = 5)]
    limit: u32,
  },
  /// Forget a record.
  Delete {
    /// The record's id.
    id: String,
    /// The record's tenant; required if it was stored under one.
    #[arg(long)]
    tenant: Option<String>,
  },
}

fn print_evaluation(response: &pb::EvaluateResponse) {
  println!(
    "verdict   {:?} (stored: {}, revision {})",
    response.verdict(),
    response.stored,
    response.revision
  );
  if let Some(similarity) = response.similarity {
    println!(
      "overall   {similarity:.3} against a threshold of {:.3}",
      response.threshold
    );
  }
  if let Some(anchor) = response.anchor_similarity {
    println!("drift     {anchor:.3} similarity to the original record");
  }
  for field in &response.fields {
    let similarity = field
      .similarity
      .map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"));
    let floor = if field.below_floor { "  << below floor" } else { "" };
    println!(
      "field     {:<14} {similarity:>6}  {:?}{floor}",
      field.field,
      field.change()
    );
  }
  for quantity in &response.quantities {
    let factor = match quantity.factor {
      Some(factor) => format!("x{factor:.2}"),
      None if quantity.zero_crossing => "no limit".to_owned(),
      None => "-".to_owned(),
    };
    let limit = if quantity.beyond_limit { "  << beyond limit" } else { "" };
    println!(
      "quantity  {:<16} {factor:>8}  {:?} ({} to {}){limit}",
      quantity.field,
      quantity.change(),
      quantity.previous_unit.as_deref().unwrap_or("-"),
      quantity.unit.as_deref().unwrap_or("-")
    );
  }
  if response.measure_changed {
    println!("measures  changed");
  }
  for reason in &response.reasons {
    println!("reason    {reason}");
  }
}

fn print_description(description: pb::DescribeResponse) {
  println!(
    "model     {} ({} dimensions per field)",
    description.model_fingerprint, description.dimension
  );
  println!("threshold {:.2}", description.threshold);
  if let Some(anchor) = description.anchor_threshold {
    println!("drift     {anchor:.2}");
  }
  for field in description.fields {
    let floor = field
      .reject_below
      .map_or_else(String::new, |floor| format!("  reject below {floor:.2}"));
    let required = if field.required { "  required" } else { "" };
    println!(
      "field     {:<14} weight {:.1}{required}{floor}",
      field.name, field.weight
    );
  }
  for quantity in description.quantities {
    let mut limits = Vec::new();
    if let Some(factor) = quantity.reject_above_factor {
      limits.push(format!("reject above a factor of {factor:.2}"));
    }
    if quantity.reject_incomparable {
      limits.push("reject incomparable units".to_owned());
    }
    if limits.is_empty() {
      limits.push("reported only".to_owned());
    }
    println!("quantity  {:<16} {}", quantity.name, limits.join(", "));
  }
  if description.reject_on_measure_change {
    println!("measures  a change of measure is rejected");
  }
}

/// Reads a record file and turns it into the request message.
fn read_record(path: &Path, id: String, tenant: Option<String>) -> anyhow::Result<pb::Record> {
  let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
  let record: RecordFields = serde_json::from_str(&text)
    .context("expected a JSON object of text fields and quantities ({\"value\": 3.5, \"unit\": \"g\"})")?;
  let quantities = record
    .quantities
    .into_iter()
    .map(|(name, quantity)| {
      let quantity = pb::Quantity {
        min: quantity.min,
        max: quantity.max,
        unit: quantity.unit,
      };
      (name, quantity)
    })
    .collect();
  Ok(pb::Record {
    id: Some(id),
    tenant,
    fields: record.texts.into_iter().collect(),
    quantities,
  })
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
  let cli = Cli::parse();
  let mut client = RecordSimilarityClient::connect(cli.server.clone())
    .await
    .with_context(|| format!("cannot connect to {}", cli.server))?;

  match cli.command {
    Command::Describe => {
      print_description(client.describe(pb::DescribeRequest {}).await?.into_inner());
    }
    Command::Evaluate {
      id,
      fields,
      tenant,
      dry_run,
      force,
    } => {
      let mode = match (dry_run, force) {
        (true, _) => pb::WriteMode::DryRun,
        (_, true) => pb::WriteMode::Force,
        _ => pb::WriteMode::Enforce,
      };
      let request = pb::EvaluateRequest {
        record: Some(read_record(&fields, id, tenant)?),
        mode: mode.into(),
        ..pb::EvaluateRequest::default()
      };
      print_evaluation(&client.evaluate(request).await?.into_inner());
    }
    Command::Similar { id, limit } => {
      let request = pb::FindSimilarRequest {
        query: Some(pb::find_similar_request::Query::Id(id)),
        limit,
        ..pb::FindSimilarRequest::default()
      };
      for neighbor in client.find_similar(request).await?.into_inner().neighbors {
        println!(
          "{:.3}  {}  {}",
          neighbor.similarity,
          neighbor.id,
          neighbor.tenant.unwrap_or_default()
        );
      }
    }
    Command::Delete { id, tenant } => {
      let request = pb::DeleteRequest { id, tenant };
      let deleted = client.delete(request).await?.into_inner().deleted;
      println!("{}", if deleted { "deleted" } else { "nothing to delete" });
    }
  }
  Ok(())
}
