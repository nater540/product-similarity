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
//! The server also exposes gRPC reflection, so `grpcurl -plaintext localhost:50051 list` works too.

#![allow(clippy::print_stdout, reason = "an example's whole job is to print")]

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use similarity_proto::v1 as pb;
use similarity_proto::v1::record_similarity_client::RecordSimilarityClient;

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
  /// Evaluate a record: a JSON object of field name to text.
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
  for reason in &response.reasons {
    println!("reason    {reason}");
  }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
  let cli = Cli::parse();
  let mut client = RecordSimilarityClient::connect(cli.server.clone())
    .await
    .with_context(|| format!("cannot connect to {}", cli.server))?;

  match cli.command {
    Command::Describe => {
      let description = client.describe(pb::DescribeRequest {}).await?.into_inner();
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
    }
    Command::Evaluate {
      id,
      fields,
      tenant,
      dry_run,
      force,
    } => {
      let text = std::fs::read_to_string(&fields).with_context(|| format!("cannot read {}", fields.display()))?;
      let fields: HashMap<String, String> = serde_json::from_str(&text).context("expected a JSON object of text")?;
      let mode = match (dry_run, force) {
        (true, _) => pb::WriteMode::DryRun,
        (_, true) => pb::WriteMode::Force,
        _ => pb::WriteMode::Enforce,
      };
      let request = pb::EvaluateRequest {
        record: Some(pb::Record {
          id: Some(id),
          tenant,
          fields,
        }),
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
