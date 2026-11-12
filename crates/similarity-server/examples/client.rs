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
  /// Find the stored records a listing from elsewhere describes, best first, and whether the best is a match.
  Match {
    /// The JSON file holding the listing, under the server's field and quantity names.
    listing: PathBuf,
    /// Only consider records of this tenant.
    #[arg(long)]
    tenant: Option<String>,
    /// How many candidates to list.
    #[arg(long, default_value_t = 5)]
    limit: u32,
    /// Hold the best candidate to this in place of the server's match threshold.
    #[arg(long)]
    threshold: Option<f32>,
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
  for (index, sentence) in response.reasons.iter().enumerate() {
    println!(
      "reason    {}",
      reason_line(response.reason_details.get(index), sentence)
    );
  }
}

/// One reason as data: its code, what it is about, and the number against its limit. The sentence instead, where
/// the data is no use to this client: a server too old to send it, or one new enough to send a code this client
/// was not built to know. An unknown reason is still a reason.
fn reason_line(detail: Option<&pb::ReasonDetail>, sentence: &str) -> String {
  let known = detail.and_then(|detail| {
    let code = pb::ReasonCode::try_from(detail.code).ok()?;
    (code != pb::ReasonCode::Unspecified).then_some((detail, code))
  });
  let Some((detail, code)) = known else {
    return sentence.to_owned();
  };
  // The same string `calibrate --ndjson` writes as a `pair_reason` code.
  let name = code.as_str_name().trim_start_matches("REASON_CODE_").to_lowercase();
  let about = match &detail.subject {
    Some(subject) => format!("{name} {subject}"),
    None => name,
  };
  let numbers = match (code, detail.value, detail.limit) {
    (pb::ReasonCode::QuantityBeyondFactor, Some(factor), Some(limit)) => format!("x{factor:.2} > x{limit:.2}"),
    (pb::ReasonCode::QuantityBeyondFactor, None, Some(limit)) => format!("from or to zero (limit x{limit:.2})"),
    (_, Some(similarity), Some(limit)) => format!("{similarity:.3} < {limit:.3}"),
    _ => String::new(),
  };
  format!("{about:<28} {numbers}").trim_end().to_owned()
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
  match description.match_threshold {
    Some(threshold) => println!(
      "matching  a listing matches at {threshold:.2}; {} records are scored for each one returned",
      description.match_overfetch
    ),
    None => println!(
      "matching  no threshold: listings are ranked, not judged; {} records are scored for each one returned",
      description.match_overfetch
    ),
  }
}

fn print_match(response: &pb::MatchResponse) {
  match response.threshold {
    Some(threshold) => println!("decision  {:?} at a threshold of {threshold:.3}", response.decision()),
    None => println!("decision  {:?}: the server has no match threshold", response.decision()),
  }
  for (rank, candidate) in (1_u32..).zip(&response.candidates) {
    println!(
      "{:<9} {:.3}  {}  {}  (the search scored it {:.3})",
      rank,
      candidate.similarity,
      candidate.id,
      candidate.tenant.as_deref().unwrap_or("-"),
      candidate.search_similarity
    );
    for field in &candidate.fields {
      let similarity = field
        .similarity
        .map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"));
      // Read with the stored record as the baseline: `Added` is a field only the listing has.
      let side = match field.change() {
        pb::FieldChange::Added => "  only the listing has it",
        pb::FieldChange::Removed => "  only the stored record has it",
        _ => "",
      };
      println!("  field     {:<14} {similarity:>6}{side}", field.field);
    }
    for quantity in &candidate.quantities {
      let factor = match quantity.factor {
        Some(factor) => format!("x{factor:.2}"),
        None => "-".to_owned(),
      };
      let unit = |unit: &Option<String>| unit.clone().unwrap_or_else(|| "-".to_owned());
      // The same reading: the stored record's value is the one before, the listing's the one after.
      let what = match quantity.change() {
        pb::QuantityChange::Added => "only the listing has it".to_owned(),
        pb::QuantityChange::Removed => "only the stored record has it".to_owned(),
        change => format!(
          "{change:?} ({} in the record, {} in the listing)",
          unit(&quantity.previous_unit),
          unit(&quantity.unit)
        ),
      };
      println!("  quantity  {:<16} {factor:>8}  {what}", quantity.field);
    }
  }
}

/// Reads a record file and turns it into the request message.
fn read_record(path: &Path, id: Option<String>, tenant: Option<String>) -> anyhow::Result<pb::Record> {
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
    id,
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
        record: Some(read_record(&fields, Some(id), tenant)?),
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
    Command::Match {
      listing,
      tenant,
      limit,
      threshold,
    } => {
      let request = pb::MatchRequest {
        // A listing carries no id and no tenant of its own; the tenant here narrows the search.
        listing: Some(read_record(&listing, None, None)?),
        limit,
        tenant,
        threshold,
      };
      print_match(&client.r#match(request).await?.into_inner());
    }
    Command::Delete { id, tenant } => {
      let request = pb::DeleteRequest { id, tenant };
      let deleted = client.delete(request).await?.into_inner().deleted;
      println!("{}", if deleted { "deleted" } else { "nothing to delete" });
    }
  }
  Ok(())
}
