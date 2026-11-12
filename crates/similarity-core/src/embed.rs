//! Text in, unit vectors out.
//!
//! [`Embedder`] owns the tokenizer and the encoder and is the only type the rest of the workspace needs in order to
//! turn strings into comparable vectors. It is synchronous and CPU-bound on purpose: the gRPC server decides how to
//! schedule it (a dedicated worker thread with micro-batching), and tests and the CLI call it directly.

use std::path::Path;

use burn::tensor::Device;

use crate::error::{Error, Result};
use crate::model::SentenceEncoder;
use crate::tokenize::{TextTokenizer, pad_batch};
use crate::vector::normalize;

/// What runs a model's forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineChoice {
  /// Burn, in Rust, from the directory's weights. Always available, and the definition of what a model computes.
  #[default]
  Burn,
  /// ONNX Runtime (C++), from `onnx/model.onnx` in the directory. Several times faster on a CPU. It needs a
  /// binary built with the `onnx` feature and a directory `tools/export_onnx.py` has been run on, and it is
  /// checked against Burn before it is used.
  Onnx,
}

impl EngineChoice {
  /// The engine's name, as the configuration and the logs spell it.
  pub fn as_str(self) -> &'static str {
    match self {
      Self::Burn => "burn",
      Self::Onnx => "onnx",
    }
  }
}

/// The texts both engines are run on before the ONNX one is trusted: short and long, so that padding, the mask,
/// and pooling are all exercised, and in one batch, so that they are padded against each other.
#[cfg(feature = "onnx")]
const PROBES: [&str; 4] = [
  "Blueberry Gummies 100mg",
  "Edibles > Gummies",
  "Ten vegan gummies infused with hybrid enhanced botanical terpenes. 10mg THC per piece, 100mg per package. Made \
   with real fruit and natural flavoring. Keep out of reach of children.",
  "a",
];
/// How far a component of a unit vector may differ between the two engines. They run the same arithmetic in
/// another order, which on the default model moves a component by two ten-millionths at most; a graph exported
/// from other weights moves them by far more.
#[cfg(feature = "onnx")]
const ENGINE_TOLERANCE: f32 = 1e-5;

#[derive(Debug)]
enum Engine {
  // Boxed: there is one engine in an embedder, and the two are very different sizes.
  Burn {
    encoder: Box<SentenceEncoder>,
    device: Device,
  },
  #[cfg(feature = "onnx")]
  Onnx(crate::model::onnx::OnnxEncoder),
}

/// Knobs that trade accuracy, latency, and storage against each other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedderOptions {
  /// Texts longer than this many tokens (special tokens included) are truncated. Attention cost grows with the
  /// square of this, and product descriptions rarely say anything new after the first couple of hundred tokens.
  pub max_tokens: usize,
  /// The most texts encoded in one forward pass.
  pub batch_size: usize,
  /// Keep only the first N components of each vector, then re-normalize. Only meaningful for models trained with
  /// Matryoshka representation learning (the default model is); `None` keeps the model's native width.
  pub output_dim: Option<usize>,
}

impl Default for EmbedderOptions {
  fn default() -> Self {
    Self {
      max_tokens: 256,
      batch_size: 32,
      output_dim: None,
    }
  }
}

/// A loaded embedding model.
#[derive(Debug)]
pub struct Embedder {
  tokenizer: TextTokenizer,
  engine: Engine,
  /// The width the model itself produces, of which `dimension` components are kept.
  native: usize,
  batch_size: usize,
  dimension: usize,
  digest: String,
  fingerprint: String,
}

impl Embedder {
  /// Loads a sentence-transformers model directory onto `device`.
  ///
  /// # Errors
  ///
  /// Returns whatever [`SentenceEncoder::load`] returns, [`Error::Tokenizer`] if `tokenizer.json` is missing or
  /// invalid, and [`Error::UnsupportedModel`] if `options` ask for something the model cannot do (more tokens than
  /// its position table holds, or more output dimensions than it produces).
  pub fn load(dir: &Path, device: Device, options: &EmbedderOptions) -> Result<Self> {
    let encoder = SentenceEncoder::load(dir, &device)?;
    let tokenizer = TextTokenizer::load(dir, options.max_tokens)?;
    let digest = encoder.files().digest()?;

    Self::assemble(tokenizer, encoder, device, options, &digest, dir)
  }

  /// Loads a model directory to run on `engine`.
  ///
  /// With [`EngineChoice::Burn`] this is [`Self::load`]. With [`EngineChoice::Onnx`] the directory's weights are
  /// loaded in Burn all the same, a set of texts is embedded by both, and the ONNX graph is used only if every
  /// component of every vector agrees: the graph is a second copy of the model that nothing else ties to the
  /// weights beside it. The Burn encoder is dropped once it has vouched for the graph. The fingerprint is the
  /// same under either engine, because the vectors are.
  ///
  /// # Errors
  ///
  /// As [`Self::load`], and [`Error::UnsupportedModel`] if the ONNX engine is asked for in a build without the
  /// `onnx` feature, if the directory has no `onnx/model.onnx`, or if that file does not compute what the
  /// directory's weights compute.
  pub fn load_with(dir: &Path, device: Device, options: &EmbedderOptions, engine: EngineChoice) -> Result<Self> {
    // Said before anything is loaded: it is a fact about the binary, not about the directory.
    #[cfg(not(feature = "onnx"))]
    if engine == EngineChoice::Onnx {
      return Err(Error::unsupported(
        dir,
        "the ONNX engine needs a binary built with `--features onnx`; this one runs models with Burn only",
      ));
    }
    let burn = Self::load(dir, device, options)?;
    match engine {
      EngineChoice::Burn => Ok(burn),
      #[cfg(feature = "onnx")]
      EngineChoice::Onnx => burn.onto_onnx(dir),
      #[cfg(not(feature = "onnx"))]
      EngineChoice::Onnx => Ok(burn),
    }
  }

  /// The same embedder with its forward pass on ONNX Runtime, if the directory's graph agrees with it.
  #[cfg(feature = "onnx")]
  fn onto_onnx(self, dir: &Path) -> Result<Self> {
    let expected = self.embed(&PROBES)?;
    let onnx = Self {
      engine: Engine::Onnx(crate::model::onnx::OnnxEncoder::load(dir, self.native)?),
      ..self
    };
    let found = onnx.embed(&PROBES)?;
    let worst = expected
      .iter()
      .flatten()
      .zip(found.iter().flatten())
      .map(|(expected, found)| (expected - found).abs())
      .fold(0.0_f32, f32::max);
    if worst > ENGINE_TOLERANCE || worst.is_nan() {
      return Err(Error::unsupported(
        dir,
        format!(
          "`{}` does not compute what the directory's weights compute (a component differs by {worst:.1e}, and \
           {ENGINE_TOLERANCE:.0e} is allowed). It was exported from other weights: run `just export-onnx <dir>` again",
          crate::model::onnx::ONNX_FILE
        ),
      ));
    }
    Ok(onnx)
  }

  /// Which engine runs this embedder's forward pass.
  pub fn engine(&self) -> EngineChoice {
    match self.engine {
      Engine::Burn { .. } => EngineChoice::Burn,
      #[cfg(feature = "onnx")]
      Engine::Onnx(_) => EngineChoice::Onnx,
    }
  }

  fn assemble(
    tokenizer: TextTokenizer,
    encoder: SentenceEncoder,
    device: Device,
    options: &EmbedderOptions,
    digest: &str,
    origin: &Path,
  ) -> Result<Self> {
    if options.batch_size == 0 {
      return Err(Error::unsupported(origin, "`batch_size` must be positive"));
    }
    if options.max_tokens > encoder.max_positions() {
      return Err(Error::unsupported(
        origin,
        format!(
          "`max_tokens` is {} but the model's position table only covers {}",
          options.max_tokens,
          encoder.max_positions()
        ),
      ));
    }
    let dimension = match options.output_dim {
      None => encoder.dimension(),
      Some(wanted) if wanted > 0 && wanted <= encoder.dimension() => wanted,
      Some(wanted) => {
        return Err(Error::unsupported(
          origin,
          format!(
            "`output_dim` is {wanted} but the model produces {} dimensions",
            encoder.dimension()
          ),
        ));
      }
    };

    // Everything that changes the numbers is in the fingerprint, so vectors stored under one fingerprint are never
    // compared against vectors computed under another.
    let fingerprint = format!("{digest}-d{dimension}-t{}", options.max_tokens);
    let embedder = Self {
      tokenizer,
      native: encoder.dimension(),
      engine: Engine::Burn {
        encoder: Box::new(encoder),
        device,
      },
      batch_size: options.batch_size,
      dimension,
      digest: digest.to_owned(),
      fingerprint,
    };
    // One real forward pass before anyone depends on this embedder. A model can load cleanly and still be unable
    // to run (a dtype or shape problem only the backend notices), and a server should find that out while it is
    // starting, not on its first request. It also pays the backend's one-time warm-up cost up front.
    embedder.embed(&["warm-up"])?;
    Ok(embedder)
  }

  /// Builds an embedder around an in-memory encoder and tokenizer, for tests that must not touch real weights.
  #[cfg(any(test, feature = "testing"))]
  pub(crate) fn from_parts(
    tokenizer: tokenizers::Tokenizer,
    encoder: SentenceEncoder,
    device: Device,
    options: &EmbedderOptions,
  ) -> Result<Self> {
    let tokenizer = TextTokenizer::new(tokenizer, options.max_tokens)?;
    Self::assemble(tokenizer, encoder, device, options, "test", Path::new("<memory>"))
  }

  /// Embeds each text into a unit-length vector of [`Self::dimension`] components, in input order.
  ///
  /// Texts are grouped by token count before batching, so one long description does not force a batch of short
  /// names to be padded to its length (see [`length_buckets`]).
  ///
  /// # Errors
  ///
  /// Returns [`Error::Tokenizer`] if a text cannot be encoded and [`Error::Inference`] if the backend returns
  /// something other than one finite vector per text.
  pub fn embed<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
    let mut encoded = Vec::with_capacity(texts.len());
    for (index, text) in texts.iter().enumerate() {
      let ids = self.tokenizer.encode(text.as_ref()).map_err(|error| match error {
        Error::Tokenizer(detail) => Error::Tokenizer(format!("text {index}: {detail}")),
        other => other,
      })?;
      encoded.push((index, ids));
    }
    encoded.sort_by_key(|(_, ids)| ids.len());

    let mut vectors: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
    for chunk in length_buckets(&encoded, self.batch_size) {
      let rows = self.forward(chunk)?;
      for ((index, _), row) in chunk.iter().zip(rows) {
        if let Some(slot) = vectors.get_mut(*index) {
          *slot = row;
        }
      }
    }
    Ok(vectors)
  }

  fn forward(&self, chunk: &[(usize, Vec<u32>)]) -> Result<Vec<Vec<f32>>> {
    let rows: Vec<&[u32]> = chunk.iter().map(|(_, ids)| ids.as_slice()).collect();
    let native = self.native;
    let flat = match &self.engine {
      Engine::Burn { encoder, device } => {
        let Some((ids, padding)) = pad_batch(&rows, device) else {
          return Ok(Vec::new());
        };
        encoder
          .forward(ids, padding)
          .into_data()
          .convert::<f32>()
          .try_into_vec::<f32>()
          .map_err(|source| Error::Inference(format!("backend returned unreadable tensor data: {source:?}")))?
      }
      #[cfg(feature = "onnx")]
      Engine::Onnx(encoder) => encoder.forward(&rows)?,
    };
    if flat.is_empty() && rows.iter().all(|row| row.is_empty()) {
      return Ok(Vec::new());
    }
    if flat.len() != chunk.len().saturating_mul(native) {
      return Err(Error::Inference(format!(
        "backend returned {} values for {} texts of dimension {native}",
        flat.len(),
        chunk.len()
      )));
    }

    flat
      .chunks(native)
      .map(|row| {
        let mut vector: Vec<f32> = row.iter().take(self.dimension).copied().collect();
        if normalize(&mut vector) {
          Ok(vector)
        } else {
          Err(Error::Inference(
            "the model produced a zero or non-finite embedding".to_owned(),
          ))
        }
      })
      .collect()
  }

  /// The number of components in every vector this embedder returns.
  pub fn dimension(&self) -> usize {
    self.dimension
  }

  /// The truncation limit, in tokens.
  pub fn max_tokens(&self) -> usize {
    self.tokenizer.max_tokens()
  }

  /// Identifies the exact model, tokenizer, output width, and truncation limit behind this embedder's vectors.
  ///
  /// Two vectors are only comparable if they were produced under the same fingerprint. It is stored next to every
  /// persisted vector and checked on the way back out.
  pub fn fingerprint(&self) -> &str {
    &self.fingerprint
  }

  /// The digest of the model's files alone: the part of the fingerprint that says which model this is, without
  /// the width and token limit it is being run at.
  pub fn content_digest(&self) -> &str {
    &self.digest
  }
}

// The gRPC server moves the embedder onto a worker thread and shares it behind an `Arc`; fail the build here, next
// to the type, if a dependency upgrade ever makes that impossible.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync + 'static>() {}
  assert_send_sync::<Embedder>();
};

/// Splits length-sorted encodings into batches of similar length.
///
/// Every row of a batch is padded to the longest row, and the encoder's cost follows the padded size, so a batch of
/// three 8-token names and one 70-token description costs as much as four descriptions. A bucket is therefore closed
/// not only when it is full but also when the next text is more than twice as long as the bucket's shortest (plus a
/// few tokens of slack, so that 3-token and 9-token texts still share a bucket instead of paying per-call overhead
/// twice). On the default model this takes a typical four-field record from about 31 ms to about 20 ms on an Apple-silicon CPU.
fn length_buckets(sorted: &[(usize, Vec<u32>)], batch_size: usize) -> Vec<&[(usize, Vec<u32>)]> {
  const SLACK_TOKENS: usize = 8;

  let mut buckets = Vec::new();
  let mut rest = sorted;
  while let Some((_, shortest)) = rest.first() {
    let limit = shortest.len().saturating_mul(2).saturating_add(SLACK_TOKENS);
    let take = rest
      .iter()
      .take(batch_size.max(1))
      .take_while(|(_, ids)| ids.len() <= limit)
      .count()
      .max(1);
    let (bucket, remainder) = rest.split_at(take.min(rest.len()));
    buckets.push(bucket);
    rest = remainder;
  }
  buckets
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::model::Pooling;
  use crate::testing::{tiny_config, tiny_embedder, tiny_tokenizer};
  use crate::vector::{cosine, norm};

  fn options(max_tokens: usize, batch_size: usize, output_dim: Option<usize>) -> EmbedderOptions {
    EmbedderOptions {
      max_tokens,
      batch_size,
      output_dim,
    }
  }

  /// The default build has no ONNX Runtime in it, and says so plainly when asked for it, before it looks at the
  /// directory at all.
  #[cfg(not(feature = "onnx"))]
  #[test]
  fn the_onnx_engine_is_refused_by_a_build_without_it() {
    let nowhere = Path::new("/no/such/model");
    let error =
      Embedder::load_with(nowhere, Device::flex(), &EmbedderOptions::default(), EngineChoice::Onnx).unwrap_err();
    assert!(error.to_string().contains("--features onnx"), "{error}");
    assert_eq!(
      (
        EngineChoice::default(),
        EngineChoice::Burn.as_str(),
        EngineChoice::Onnx.as_str()
      ),
      (EngineChoice::Burn, "burn", "onnx")
    );
  }

  #[test]
  fn the_tiny_embedder_runs_on_burn() {
    assert_eq!(tiny_embedder(&options(16, 8, None)).engine(), EngineChoice::Burn);
  }

  #[test]
  fn every_embedding_is_a_unit_vector_of_the_configured_width() {
    let embedder = tiny_embedder(&EmbedderOptions {
      max_tokens: 16,
      ..EmbedderOptions::default()
    });
    let vectors = embedder
      .embed(&["blueberry gummies", "vape cartridge", "flower"])
      .unwrap();
    assert_eq!(vectors.len(), 3);
    for vector in &vectors {
      assert_eq!(vector.len(), 16);
      assert!((norm(vector) - 1.0).abs() < 1e-5);
    }
  }

  #[test]
  fn results_come_back_in_input_order_whatever_the_batching() {
    let texts = [
      "blueberry gummies indica hybrid",
      "vape",
      "flower sativa",
      "wyld edibles gummies",
      "resin",
    ];
    // A batch size of two forces three forward passes over length-sorted chunks, so the input order is scrambled
    // internally and every text is padded against a different neighbor than it would be alone.
    let embedder = tiny_embedder(&options(16, 2, None));
    let batched = embedder.embed(&texts).unwrap();
    assert_eq!(batched.len(), texts.len());
    for (text, expected) in texts.iter().zip(&batched) {
      let alone = embedder.embed(&[*text]).unwrap();
      assert!(cosine(&alone[0], expected).unwrap() > 0.9999, "{text}");
    }
  }

  #[test]
  fn text_beyond_the_token_limit_is_truncated_not_rejected() {
    let embedder = tiny_embedder(&options(6, 32, None));
    let long = "gummies ".repeat(100);
    // Six tokens: [CLS], four words, [SEP].
    let short = "gummies gummies gummies gummies";
    let vectors = embedder.embed(&[long.as_str(), short]).unwrap();
    assert!(cosine(&vectors[0], &vectors[1]).unwrap() > 0.9999);
  }

  #[test]
  fn buckets_respect_the_batch_size_and_keep_long_texts_away_from_short_ones() {
    let lengths = [3_usize, 4, 9, 14, 15, 70, 71, 200];
    let sorted: Vec<(usize, Vec<u32>)> = lengths.iter().enumerate().map(|(i, len)| (i, vec![0; *len])).collect();

    let shapes = |batch_size| -> Vec<Vec<usize>> {
      length_buckets(&sorted, batch_size)
        .iter()
        .map(|bucket| bucket.iter().map(|(_, ids)| ids.len()).collect())
        .collect()
    };
    // 3 admits up to 14 tokens; 15 admits up to 38; 70 admits up to 148; 200 is alone.
    assert_eq!(shapes(32), [vec![3, 4, 9, 14], vec![15], vec![70, 71], vec![200]]);
    assert_eq!(shapes(2), [vec![3, 4], vec![9, 14], vec![15], vec![70, 71], vec![200]]);
    assert!(length_buckets(&[], 32).is_empty());
  }

  #[test]
  fn an_empty_batch_is_an_empty_result() {
    let embedder = tiny_embedder(&EmbedderOptions {
      max_tokens: 16,
      ..EmbedderOptions::default()
    });
    assert!(embedder.embed::<&str>(&[]).unwrap().is_empty());
  }

  #[test]
  fn output_dim_truncates_and_renormalizes() {
    let embedder = tiny_embedder(&options(16, 32, Some(4)));
    let vectors = embedder.embed(&["blueberry gummies"]).unwrap();
    assert_eq!(embedder.dimension(), 4);
    assert_eq!(vectors[0].len(), 4);
    assert!((norm(&vectors[0]) - 1.0).abs() < 1e-5);
  }

  #[test]
  fn impossible_options_are_refused_at_load() {
    let device = Device::flex();
    let build = |options: &EmbedderOptions| {
      let config = tiny_config();
      let encoder = SentenceEncoder::from_parts(config.init(&device), config, Pooling::Mean);
      Embedder::from_parts(tiny_tokenizer(), encoder, device.clone(), options)
    };
    assert!(build(&options(1000, 32, None)).is_err(), "more tokens than positions");
    assert!(
      build(&options(16, 32, Some(17))).is_err(),
      "more dimensions than the model has"
    );
    assert!(build(&options(16, 32, Some(0))).is_err(), "zero dimensions");
    assert!(build(&options(16, 0, None)).is_err(), "zero batch size");
  }

  #[test]
  fn the_fingerprint_changes_with_anything_that_changes_the_vectors() {
    let base = tiny_embedder(&options(16, 32, None));
    assert_eq!(
      base.fingerprint(),
      tiny_embedder(&options(16, 8, None)).fingerprint(),
      "batch size is not identity"
    );
    assert_ne!(base.fingerprint(), tiny_embedder(&options(12, 32, None)).fingerprint());
    assert_ne!(
      base.fingerprint(),
      tiny_embedder(&options(16, 32, Some(8))).fingerprint()
    );
  }
}
