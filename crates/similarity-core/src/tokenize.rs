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

/// How many bytes of a long text the tokenizer is given at first, for every token it may keep.
///
/// A tokenizer walks the whole of a text before truncating it, so the cost of a field grows with its length even
/// though only the first `max_tokens` tokens are ever used: a megabyte of text took most of a second, on the one
/// thread every request shares. A long text is therefore handed over a part at a time (see
/// [`TextTokenizer::encode`]), and this sets the size of the first part. A token of ordinary text is four or five
/// bytes, so the first part nearly always holds every token that will be kept.
const BYTES_PER_TOKEN: usize = 64;

/// How many times longer the next part of a text is, when the part before it did not fill the token limit.
///
/// The parts tried add up to at most a third more than the text itself, so the worst a text built to defeat the
/// first part can do is cost a third more than it did before parts existed.
const GROWTH: usize = 4;

/// The shortest head of `text` that is at least `at_least` bytes long and stops just before a space, a tab, or a
/// line break; the whole text when there is no such place.
///
/// Those four bytes separate words for every tokenizer, so the tokens of a head that stops at one are exactly the
/// first tokens of the whole text. No other character is trusted to: BERT's normalizer deletes a form feed, which
/// joins the words on either side of it, and it deletes zero-width characters outright.
fn head(text: &str, at_least: usize) -> &str {
  let gap = text.as_bytes().get(at_least..).and_then(|rest| {
    rest
      .iter()
      .position(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\r'))
  });
  // An ASCII byte is never part of a longer character, so this is a character boundary.
  gap
    .and_then(|offset| text.get(..at_least.saturating_add(offset)))
    .unwrap_or(text)
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
  /// The result is what the tokenizer makes of the whole text, but a long text is not read to its end to get it.
  /// The tokenizer is given a head of the text that stops between two words, and that is the answer as soon as it
  /// fills the limit: whatever follows could only have been truncated away. A head that does not fill the limit
  /// says nothing about what follows it, so a longer one is tried, up to the whole text. How many bytes it takes
  /// to fill the limit is never guessed: sixteen kilobytes of zero-width characters, of combining accents, or of
  /// one unbroken word come to no tokens or to one, and the words after them are still read (D21).
  ///
  /// # Errors
  ///
  /// Returns [`Error::Tokenizer`] if the text cannot be encoded.
  pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
    let mut at_least = self.max_tokens.saturating_mul(BYTES_PER_TOKEN);
    loop {
      let part = head(text, at_least);
      let ids = self.encode_whole(part)?;
      if part.len() >= text.len() || ids.len() >= self.max_tokens {
        return Ok(ids);
      }
      at_least = part.len().max(1).saturating_mul(GROWTH);
    }
  }

  /// What the tokenizer makes of all of `text`, read to its end. The reference [`Self::encode`] must agree with.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Tokenizer`] if the text cannot be encoded.
  pub(crate) fn encode_whole(&self, text: &str) -> Result<Vec<u32>> {
    let encoding = self
      .inner
      .encode_fast(text, true)
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
  use tokenizers::normalizers::BertNormalizer;

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

  /// The tiny tokenizer with BERT's normalizer in front of it, which deletes control and format characters and
  /// strips accents, as the real model's does.
  fn bert_like(max_tokens: usize) -> TextTokenizer {
    let mut tokenizer = tiny_tokenizer();
    let _ = tokenizer.with_normalizer(Some(BertNormalizer::default()));
    TextTokenizer::new(tokenizer, max_tokens).unwrap()
  }

  #[test]
  fn a_long_text_is_cut_only_between_two_words() {
    assert_eq!(
      head("blueberry gummies", 64),
      "blueberry gummies",
      "a short text is whole"
    );
    assert_eq!(
      head("blueberry gummies", 0),
      "blueberry",
      "the first gap at or after the mark"
    );
    assert_eq!(head("blueberry gummies", 9), "blueberry");
    assert_eq!(
      head("blueberry gummies", 10),
      "blueberry gummies",
      "no gap is left, so all of it"
    );
    assert_eq!(head("blueberry\tgummies\nvape\rcartridge", 10), "blueberry\tgummies");
    // A form feed, a no-break space, and a zero-width space are not gaps every tokenizer agrees on.
    for joined in [
      "blueberry\u{c}gummies",
      "blueberry\u{a0}gummies",
      "blueberry\u{200b}gummies",
    ] {
      assert_eq!(head(joined, 0), joined);
    }
    // The mark may fall inside a character; the cut never does.
    let accented = format!("{} gummies", "é".repeat(40));
    assert_eq!(head(&accented, 41), "é".repeat(40));
    assert_eq!(head("", 0), "");
  }

  #[test]
  fn a_form_feed_joins_two_words_for_a_bert_tokenizer_so_it_is_no_place_to_cut() {
    let tokenizer = bert_like(8);
    assert_eq!(tokenizer.encode("blueberry gummies").unwrap(), [2, 4, 5, 3]);
    assert_eq!(
      tokenizer.encode("blueberry\u{c}gummies").unwrap(),
      [2, 1, 3],
      "one unknown word"
    );
  }

  /// D21 as first built cut a text at a number of bytes, and these all slipped the real text past the cut: two
  /// different products, padded alike, embedded identically and scored 1.0.
  #[test]
  fn padding_cannot_push_a_text_out_of_what_is_read() {
    let tokenizer = bert_like(4);
    let limit = 4 * BYTES_PER_TOKEN;
    let paddings = [
      ("zero-width spaces", "\u{200b}".repeat(limit), vec![2, 4, 5, 3]),
      ("word joiners", "\u{2060}".repeat(limit), vec![2, 4, 5, 3]),
      ("combining accents", "\u{301}".repeat(limit), vec![2, 4, 5, 3]),
      (
        "one unbroken word",
        format!("{} ", "a".repeat(limit * 3)),
        vec![2, 1, 4, 3],
      ),
      ("spaces", " ".repeat(limit * 3), vec![2, 4, 5, 3]),
    ];
    for (name, padding, expected) in paddings {
      let text = format!("{padding}blueberry gummies");
      assert_eq!(tokenizer.encode(&text).unwrap(), expected, "{name}");
      assert_eq!(
        tokenizer.encode_whole(&text).unwrap(),
        expected,
        "{name}: the reference"
      );
      let other = tokenizer.encode(&format!("{padding}vape cartridge")).unwrap();
      assert_ne!(other, expected, "{name}: another product is another row of tokens");
    }
  }

  #[test]
  fn a_very_long_text_encodes_to_the_same_tokens_as_reading_all_of_it() {
    let tokenizer = TextTokenizer::new(tiny_tokenizer(), 16).unwrap();
    let megabyte = "blueberry gummies vape cartridge edibles flower ".repeat(22_000);
    assert!(megabyte.len() > 1_000_000);
    let whole = tokenizer.encode_whole(&megabyte).unwrap();
    assert_eq!(whole.len(), 16);
    assert_eq!(tokenizer.encode(&megabyte).unwrap(), whole);

    // Unbroken text has no gap to stop at, so all of it is read.
    let unbroken = "a".repeat(1_000_000);
    assert_eq!(tokenizer.encode(&unbroken).unwrap(), [2, 1, 3], "[CLS] [UNK] [SEP]");
  }

  #[test]
  fn a_long_ordinary_text_is_not_read_to_its_end() {
    // The tokenizer refuses nothing, so reading is observed through what a head has to contain: the first part of
    // an ordinary text already fills the limit, and the head is all that `encode` hands over.
    let tokenizer = TextTokenizer::new(tiny_tokenizer(), 4).unwrap();
    let long = "blueberry gummies ".repeat(1000);
    let first = head(&long, 4 * BYTES_PER_TOKEN);
    assert!(first.len() < 4 * BYTES_PER_TOKEN + "blueberry".len() + 1);
    assert_eq!(
      tokenizer.encode_whole(first).unwrap().len(),
      4,
      "the first part fills the limit"
    );
    assert_eq!(tokenizer.encode(&long).unwrap(), tokenizer.encode_whole(first).unwrap());
  }
}
