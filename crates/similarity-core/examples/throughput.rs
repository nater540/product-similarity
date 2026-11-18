//! A CPU throughput probe: how long does one forward pass take for the batch shapes the server actually sees?
//!
//! ```text
//! cargo run --release -p similarity-core --example throughput -- models/mdbr-leaf-mt
//! cargo run --release -p similarity-core --features onnx --example throughput -- models/mdbr-leaf-mt onnx
//! cargo run --release -p similarity-core --features onnx --example throughput -- models/mdbr-leaf-mt onnx 6
//! ```
//!
//! The shapes mirror real traffic: a single short field (one edited name), one whole record (four fields, one of
//! them a description), and a full micro-batch. Run it before changing `max_tokens`, `batch_size`, `threads`, the
//! model, or the backend features, because those are the knobs that move these numbers.

#![allow(clippy::print_stdout, reason = "an example's whole job is to print")]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use similarity_core::{Device, Embedder, EmbedderOptions, EngineChoice};

const NAME: &str = "Blueberry Gummies 100mg";
const CATEGORY: &str = "Edibles > Gummies";
const BRAND: &str = "WYLD";
const DESCRIPTION: &str = "Ten vegan gummies infused with hybrid enhanced botanical terpenes. 10mg THC per piece, \
                           100mg per package. Made with real fruit and natural flavoring. Keep out of reach of \
                           children. Store in a cool, dry place away from direct sunlight.";

fn median(mut samples: Vec<Duration>) -> Duration {
  samples.sort();
  samples.get(samples.len() / 2).copied().unwrap_or_default()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let dir = PathBuf::from(
    std::env::args()
      .nth(1)
      .ok_or("usage: throughput <model-dir> [burn|onnx] [threads]")?,
  );
  let engine = match std::env::args().nth(2).as_deref() {
    None | Some("burn") => EngineChoice::Burn,
    Some("onnx") => EngineChoice::Onnx,
    Some(other) => return Err(format!("`{other}` is not an engine; it is `burn` or `onnx`").into()),
  };
  // As `model.threads`: how many threads ONNX Runtime spreads a pass across, left to it when not given.
  let threads = match std::env::args().nth(3) {
    None => None,
    Some(given) => Some(
      given
        .parse::<usize>()
        .map_err(|_| format!("`{given}` is not a number of threads"))?,
    ),
  };
  let options = EmbedderOptions {
    threads,
    ..EmbedderOptions::default()
  };
  let started = Instant::now();
  let embedder = Embedder::load_with(&dir, Device::flex(), &options, engine)?;
  println!(
    "loaded {} in {:.0?} (dimension {}, engine {})",
    dir.display(),
    started.elapsed(),
    embedder.dimension(),
    embedder.engine().as_str()
  );

  let record = [NAME, CATEGORY, BRAND, DESCRIPTION];
  let shapes: [(&str, Vec<&str>); 4] = [
    ("1 short field", vec![NAME]),
    ("1 description", vec![DESCRIPTION]),
    ("1 record (4 fields)", record.to_vec()),
    (
      "8 records (32 fields)",
      record.iter().cycle().take(32).copied().collect(),
    ),
  ];

  for (label, texts) in &shapes {
    embedder.embed(texts)?; // warm-up
    let samples = (0..20)
      .map(|_| {
        let started = Instant::now();
        embedder.embed(texts).map(|_| started.elapsed())
      })
      .collect::<Result<Vec<_>, _>>()?;
    let median = median(samples);
    #[expect(clippy::cast_precision_loss, reason = "a handful of texts")]
    let per_second = texts.len() as f64 / median.as_secs_f64();
    println!("{label:<24} median {median:>10.2?}   {per_second:>8.0} texts/s");
  }
  Ok(())
}
