//! Text to token ids, and token ids to padded batch tensors.
//!
//! Both halves are shared by [`crate::Embedder`] and by training (`similarity-train`), so that a model is fine-tuned
//! on exactly the token sequences it is later served with. A second copy of "how texts are truncated and padded"
//! would be free to drift, and that kind of drift does not fail: it only makes the served model slightly worse than
//! the one that was evaluated.

use std::path::Path;

use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use tokenizers::{Tokenizer, TruncationParams};

use crate::error::{Error, Result};

/// The file a model directory keeps its tokenizer in.
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// How many bytes of a text are handed to the tokenizer for every token it is allowed to keep.
///
/// A tokenizer walks the whole of a text before truncating it, so the cost of a field grows with its length even
/// though only the first `max_tokens` tokens are ever used: a megabyte of text took most of a second, on the one
/// thread every request shares. A token of ordinary text is four or five bytes, and no vocabulary entry is
/// anywhere near this long, so the first `max_tokens` tokens of a text lie well inside its first
/// `max_tokens * BYTES_PER_TOKEN` bytes and what is cut off was never going to be read. (The exception is text
/// made of unbroken runs longer than this, which is a row of unknown tokens with or without the cut.)
const BYTES_PER_TOKEN: usize = 64;

/// The part of `text` worth tokenizing when at most `max_tokens` tokens are kept: all of it, unless it is longer
/// than any text that could need more. The cut falls on a character boundary.
fn head(text: &str, max_tokens: usize) -> &str {
  let limit = max_tokens.saturating_mul(BYTES_PER_TOKEN);
  if text.len() <= limit {
    return text;
  }
  // A UTF-8 character is at most four bytes, so a boundary is never more than three steps back.
  let end = (0..=limit).rev().find(|end| text.is_char_boundary(*end)).unwrap_or(0);
  text.get(..end).unwrap_or(text)
}

/// A model's tokenizer with this crate's truncation and padding policy applied.
#[derive(Debug, Clone)]
pub struct TextTokenizer {
  inner: Tokenizer,
  max_tokens: usize,
}

impl TextTokenizer {
  /// Loads `dir/tokenizer.json`, truncating to `max_tokens` (special tokens included).
  ///
  /// # Errors
  ///
  /// Returns [`Error::Tokenizer`] if the file is missing or invalid, or if it cannot truncate to `max_tokens`.
  pub fn load(dir: &Path, max_tokens: usize) -> Result<Self> {
    let path = dir.join(TOKENIZER_FILE);
    let tokenizer = Tokenizer::from_file(&path)
      .map_err(|source| Error::Tokenizer(format!("cannot load {}: {source}", path.display())))?;
    Self::new(tokenizer, max_tokens)
  }

  /// Applies the truncation and padding policy to an already-built tokenizer.
  ///
  /// A `tokenizer.json` carries whatever truncation and padding its exporter happened to use (the default model's
  /// says "pad everything to 128"). Both are replaced: truncation by `max_tokens`, padding by the per-batch padding
  /// of [`pad_batch`].
  ///
  /// # Errors
  ///
  /// Returns [`Error::Tokenizer`] if the tokenizer cannot truncate to `max_tokens`.
  pub(crate) fn new(mut tokenizer: Tokenizer, max_tokens: usize) -> Result<Self> {
    tokenizer
      .with_truncation(Some(TruncationParams {
        max_length: max_tokens,
        ..TruncationParams::default()
      }))
      .map_err(|source| Error::Tokenizer(format!("cannot truncate to {max_tokens} tokens: {source}")))?;
    tokenizer.with_padding(None);
    Ok(Self {
      inner: tokenizer,
      max_tokens,
    })
  }

  /// The token ids of `text`, special tokens included, at most [`Self::max_tokens`] of them.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Tokenizer`] if the text cannot be encoded.
  pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
    let encoding = self
      .inner
      .encode_fast(head(text, self.max_tokens), true)
      .map_err(|source| Error::Tokenizer(format!("cannot encode text: {source}")))?;
    Ok(encoding.get_ids().to_vec())
  }

  /// The truncation limit, in tokens.
  pub fn max_tokens(&self) -> usize {
    self.max_tokens
  }
}

/// Packs token rows of unequal length into one `[rows, longest]` id tensor and its padding mask.
///
/// The mask is `true` at padded positions. Returns `None` when there is nothing to encode (no rows, or only empty
/// ones), since a zero-width tensor is not something the encoder can run.
pub fn pad_batch(rows: &[&[u32]], device: &Device) -> Option<(Tensor<2, Int>, Tensor<2, Bool>)> {
  let width = rows.iter().map(|row| row.len()).max().unwrap_or(0);
  if width == 0 {
    return None;
  }
  let cells = rows.len().saturating_mul(width);
  let mut ids = Vec::with_capacity(cells);
  let mut padding = Vec::with_capacity(cells);
  for row in rows {
    let pad = width.saturating_sub(row.len());
    ids.extend(row.iter().map(|id| i64::from(*id)));
    // The value at padded positions is irrelevant (they are masked out of attention and pooling); zero is BERT's
    // own `[PAD]` id, which keeps a debugger's view of the batch unsurprising.
    ids.extend(std::iter::repeat_n(0_i64, pad));
    padding.extend(std::iter::repeat_n(false, row.len()));
    padding.extend(std::iter::repeat_n(true, pad));
  }

  let shape = [rows.len(), width];
  Some((
    Tensor::<2, Int>::from_data(TensorData::new(ids, shape), device),
    Tensor::<2, Bool>::from_data(TensorData::new(padding, shape), device),
  ))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::tiny_tokenizer;

  #[test]
  fn rows_are_padded_to_the_longest_and_the_mask_marks_the_padding() {
    let device = Device::flex();
    let (ids, padding) = pad_batch(&[&[2, 4, 5, 3], &[2, 3]], &device).unwrap();
    assert_eq!(ids.dims(), [2, 4]);
    assert_eq!(
      ids.into_data().convert::<i64>().try_into_vec::<i64>().unwrap(),
      [2, 4, 5, 3, 2, 3, 0, 0]
    );
    assert_eq!(
      padding.into_data().try_into_vec::<bool>().unwrap(),
      [false, false, false, false, false, false, true, true]
    );
  }

  #[test]
  fn nothing_to_encode_is_not_a_batch() {
    let device = Device::flex();
    assert!(pad_batch(&[], &device).is_none());
    assert!(pad_batch(&[&[], &[]], &device).is_none());
  }

  #[test]
  fn encoding_adds_special_tokens_and_truncates_to_the_limit() {
    let tokenizer = TextTokenizer::new(tiny_tokenizer(), 4).unwrap();
    // [CLS] blueberry gummies [SEP]
    assert_eq!(tokenizer.encode("blueberry gummies").unwrap(), [2, 4, 5, 3]);
    let long = tokenizer.encode("blueberry gummies vape cartridge").unwrap();
    assert_eq!(long.len(), 4);
    assert_eq!(tokenizer.max_tokens(), 4);
  }

  #[test]
  fn only_the_part_of_a_text_that_could_be_read_is_tokenized() {
    assert_eq!(
      head("blueberry gummies", 4),
      "blueberry gummies",
      "a short text is handed over whole"
    );
    let long = "blueberry gummies ".repeat(1000);
    let kept = head(&long, 4);
    assert_eq!(kept.len(), 4 * BYTES_PER_TOKEN);
    assert!(long.starts_with(kept));
    assert_eq!(head(&long, 1000), long, "a text within the limit is not cut");
    assert_eq!(head(&long, 0), "", "nothing is kept, so nothing is read");
  }

  #[test]
  fn the_cut_never_splits_a_character() {
    // Two bytes each, so an odd limit falls in the middle of one; three and four bytes each likewise.
    for (character, max_tokens) in [("é", 1), ("ブ", 1), ("🍇", 1), ("é", 3)] {
      let long = character.repeat(200);
      let kept = head(&long, max_tokens);
      let limit = max_tokens * BYTES_PER_TOKEN;
      assert!(
        kept.len() <= limit && kept.len() > limit - 4,
        "{character}: {}",
        kept.len()
      );
      assert_eq!(
        kept.len() % character.len(),
        0,
        "{character}: a whole number of characters"
      );
      assert!(long.starts_with(kept));
    }
  }

  #[test]
  fn a_very_long_text_encodes_to_the_same_tokens_as_before_the_cut_existed() {
    let tokenizer = TextTokenizer::new(tiny_tokenizer(), 16).unwrap();
    let megabyte = "blueberry gummies vape cartridge edibles flower ".repeat(22_000);
    assert!(megabyte.len() > 1_000_000);
    // What the tokenizer itself makes of the whole text: the reference the cut must not change.
    let whole = tokenizer
      .inner
      .encode_fast(megabyte.as_str(), true)
      .unwrap()
      .get_ids()
      .to_vec();
    assert_eq!(whole.len(), 16);
    assert_eq!(tokenizer.encode(&megabyte).unwrap(), whole);

    // Unbroken text has no word boundary to cut at, and must not be a problem either.
    let unbroken = "a".repeat(1_000_000);
    assert_eq!(tokenizer.encode(&unbroken).unwrap(), [2, 1, 3], "[CLS] [UNK] [SEP]");
  }
}
