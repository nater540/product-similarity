//! The encoder through ONNX Runtime, for a server that wants the forward pass faster than Burn's CPU backend runs it.
//!
//! This is the one place the workspace calls into C++. ONNX Runtime is not a Rust library: the `ort` crate binds
//! Microsoft's runtime, which is linked into the binary. It is behind the `onnx` feature so that the default build
//! has none of it, and it computes from a file, `onnx/model.onnx`, that `tools/export_onnx.py` writes from the
//! model directory's weights.
//!
//! **That file is not taken on trust.** It is a second copy of the model: nothing in it says which weights it was
//! made from, and a directory whose weights were fine-tuned after the export still has the old file. So an
//! [`OnnxEncoder`] is only ever put behind an [`crate::Embedder`] that has first run the same texts through the
//! directory's weights in Burn and found the two to agree (see `Embedder::load_with`). The Burn encoder stays the
//! definition of what a model computes; this is a faster way of computing it.

use std::fmt;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::AutoDevicePolicy;
use ort::value::Tensor;

use crate::error::{Error, Result};

/// Where the graph is, inside a model directory.
pub const ONNX_FILE: &str = "onnx/model.onnx";
/// The graph's output: one vector of the model's native width per text, pooled and projected, not normalized.
const OUTPUT: &str = "embedding";

/// A model directory's ONNX graph, loaded and ready to run on the CPU.
pub struct OnnxEncoder {
  // Running a session needs it exclusively. One forward pass at a time is what the server asks for anyway (D9).
  session: Mutex<Session>,
  dimension: usize,
}

impl fmt::Debug for OnnxEncoder {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("OnnxEncoder")
      .field("dimension", &self.dimension)
      .finish_non_exhaustive()
  }
}

/// ONNX Runtime allows one environment in a process. It is made the first time a graph is loaded and kept.
static ENVIRONMENT: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();

fn environment() -> Result<&'static Environment> {
  ENVIRONMENT
    .get_or_init(|| {
      ort::init()
        .with_telemetry(false)
        .build()
        .map_err(|error| error.to_string())
    })
    .as_ref()
    .map_err(|error| failed("cannot start", error))
}

fn failed(what: &str, error: impl fmt::Display) -> Error {
  Error::Inference(format!("ONNX Runtime: {what}: {error}"))
}

impl OnnxEncoder {
  /// Loads `onnx/model.onnx` from `dir`. `dimension` is the native width the directory's weights produce; the
  /// graph has to produce the same.
  ///
  /// # Errors
  ///
  /// Returns [`Error::UnsupportedModel`] if the file is not there, and [`Error::Inference`] if ONNX Runtime
  /// cannot load it.
  pub fn load(dir: &Path, dimension: usize) -> Result<Self> {
    let path = dir.join(ONNX_FILE);
    if !path.is_file() {
      return Err(Error::unsupported(
        dir,
        format!("the ONNX engine needs `{ONNX_FILE}`, which is not there; write it with `just export-onnx <dir>`"),
      ));
    }
    let session = Session::builder(environment()?)
      .map_err(|error| failed("cannot configure a session", error))?
      // On macOS the default is to hand the graph to CoreML, which is slower for a model this small and cannot run
      // every batch shape. This engine is a CPU engine everywhere.
      .with_auto_device(AutoDevicePolicy::PreferCPU)
      .map_err(|error| failed("cannot select the CPU", error))?
      .commit_from_file(&path)
      .map_err(|error| failed(&format!("cannot load {}", path.display()), error))?;
    Ok(Self {
      session: Mutex::new(session),
      dimension,
    })
  }

  /// Runs token rows of unequal length through the graph and returns one row of `dimension` values per token
  /// row, concatenated. Rows are padded to the longest; padded positions are masked.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Inference`] if the runtime fails or returns anything but `rows.len()` rows of the expected
  /// width.
  pub fn forward(&self, rows: &[&[u32]]) -> Result<Vec<f32>> {
    let width = rows.iter().map(|row| row.len()).max().unwrap_or(0);
    if width == 0 {
      return Ok(Vec::new());
    }
    let cells = rows.len().saturating_mul(width);
    let mut ids = Vec::with_capacity(cells);
    let mut mask = Vec::with_capacity(cells);
    for row in rows {
      let pad = width.saturating_sub(row.len());
      ids.extend(row.iter().map(|id| i64::from(*id)));
      ids.extend(std::iter::repeat_n(0_i64, pad));
      mask.extend(std::iter::repeat_n(1_i64, row.len()));
      mask.extend(std::iter::repeat_n(0_i64, pad));
    }
    let shape = [rows.len(), width];
    let input_ids = Tensor::from_array((shape, ids)).map_err(|error| failed("cannot build the ids", error))?;
    let attention_mask = Tensor::from_array((shape, mask)).map_err(|error| failed("cannot build the mask", error))?;

    let mut session = self
      .session
      .lock()
      .map_err(|_| Error::Inference("ONNX Runtime: an earlier forward pass panicked".to_owned()))?;
    let outputs = session
      .run(ort::inputs!["input_ids" => input_ids, "attention_mask" => attention_mask])
      .map_err(|error| failed("the forward pass failed", error))?;
    let (_, values) = outputs
      .get(OUTPUT)
      .ok_or_else(|| Error::Inference(format!("ONNX Runtime: the graph has no `{OUTPUT}` output")))?
      .try_extract_tensor::<f32>()
      .map_err(|error| failed("the output is not 32-bit floats", error))?;
    if values.len() != rows.len().saturating_mul(self.dimension) {
      return Err(Error::Inference(format!(
        "ONNX Runtime returned {} values for {} texts of dimension {}; `{ONNX_FILE}` is not this model's",
        values.len(),
        rows.len(),
        self.dimension
      )));
    }
    Ok(values.to_vec())
  }
}
