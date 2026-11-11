//! From a caller's field map to the vectors that get stored.
//!
//! The path is split in two around the (slow, batched, possibly remote) embedding call so that this crate stays
//! synchronous and the caller decides how to run the model:
//!
//! 1. [`PreparedRecord::new`] validates the fields against the schema, normalizes their text, and fingerprints it.
//! 2. [`PreparedRecord::plan`] works out which fields actually need embedding. A field whose text fingerprint matches
//!    the stored baseline reuses the baseline's block: most edits to a record touch one field, so this skips most of
//!    the model work on the hot path.
//! 3. The caller embeds [`Plan::texts`] and hands the vectors to [`Plan::finish`], which packs the composite.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::schema::Schema;
use crate::vector::normalize;

/// Collapses every run of whitespace to one space and trims the ends. Returns `None` if nothing is left.
///
/// Whitespace is the one thing normalized here because it is the one difference that is never a real edit: a
/// description re-saved with a trailing newline or re-wrapped by a rich-text editor must hash, and therefore embed,
/// identically. Case and accents are left to the model's own tokenizer.
pub fn normalize_text(raw: &str) -> Option<String> {
  let mut words = raw.split_whitespace();
  let first = words.next()?;
  let mut text = String::with_capacity(raw.len());
  text.push_str(first);
  for word in words {
    text.push(' ');
    text.push_str(word);
  }
  Some(text)
}

/// A stable fingerprint of a field's normalized text: the first 16 bytes of its SHA-256, in hex.
///
/// Used only to detect "this field did not change", never for security, so 128 bits is ample.
pub fn text_hash(normalized: &str) -> String {
  hex_prefix(&Sha256::digest(normalized.as_bytes()), 16)
}

/// The first `bytes` bytes of `digest` as lowercase hex.
pub(crate) fn hex_prefix(digest: &[u8], bytes: usize) -> String {
  use std::fmt::Write as _;
  digest.iter().take(bytes).fold(String::new(), |mut hex, byte| {
    // Writing to a `String` cannot fail.
    let _ = write!(hex, "{byte:02x}");
    hex
  })
}

/// The stored form of a record: its composite vector and the text fingerprint of every field it has a value for.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordVectors {
  /// The composite described in [`crate::schema`].
  pub composite: Vec<f32>,
  /// Field name to [`text_hash`] of that field's normalized text. Fields without a value are absent.
  pub hashes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedField {
  text: String,
  hash: String,
}

/// A record whose fields have been validated against a schema, normalized, and fingerprinted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedRecord {
  /// One entry per schema field, in schema order; `None` where the record has no value.
  fields: Vec<Option<PreparedField>>,
}

impl PreparedRecord {
  /// Validates `fields` against `schema`.
  ///
  /// A field that is missing and a field that is present but blank are the same thing: no value.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Record`] if a field is not in the schema, appears twice, a required field has no value, or no
  /// field has a value at all.
  pub fn new<K, V>(schema: &Schema, fields: impl IntoIterator<Item = (K, V)>) -> Result<Self>
  where
    K: AsRef<str>,
    V: AsRef<str>,
  {
    let mut by_name: BTreeMap<String, Option<String>> = BTreeMap::new();
    for (name, value) in fields {
      let name = name.as_ref();
      if schema.field(name).is_none() {
        let known: Vec<&str> = schema.fields().iter().map(|field| field.name.as_str()).collect();
        return Err(Error::Record(format!(
          "unknown field `{name}`; the schema has: {}",
          known.join(", ")
        )));
      }
      if by_name
        .insert(name.to_owned(), normalize_text(value.as_ref()))
        .is_some()
      {
        return Err(Error::Record(format!("field `{name}` is given twice")));
      }
    }

    let mut prepared = Vec::with_capacity(schema.fields().len());
    for spec in schema.fields() {
      let text = by_name.remove(&spec.name).flatten();
      if text.is_none() && spec.required {
        return Err(Error::Record(format!(
          "required field `{}` is missing or blank",
          spec.name
        )));
      }
      prepared.push(text.map(|text| PreparedField {
        hash: text_hash(&text),
        text,
      }));
    }
    if prepared.iter().all(Option::is_none) {
      return Err(Error::Record("no field has a value".to_owned()));
    }
    Ok(Self { fields: prepared })
  }

  /// Decides, field by field, whether to embed the text or reuse the baseline's vector.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Incompatible`] if `baseline`'s composite does not have this schema's layout.
  pub fn plan<'a>(
    &'a self,
    schema: &'a Schema,
    dimension: usize,
    baseline: Option<&'a RecordVectors>,
  ) -> Result<Plan<'a>> {
    let baseline_blocks = match baseline {
      Some(stored) => schema.blocks(dimension, &stored.composite)?,
      None => vec![None; schema.fields().len()],
    };

    let slots = self
      .fields
      .iter()
      .zip(schema.fields())
      .zip(baseline_blocks)
      .map(|((prepared, spec), stored_block)| {
        let Some(prepared) = prepared else {
          return Slot::Absent;
        };
        let stored_hash = baseline.and_then(|stored| stored.hashes.get(&spec.name));
        match stored_block {
          Some(block) if stored_hash == Some(&prepared.hash) => Slot::Reuse {
            name: &spec.name,
            hash: &prepared.hash,
            block,
          },
          _ => Slot::Embed {
            name: &spec.name,
            hash: &prepared.hash,
            text: &prepared.text,
          },
        }
      })
      .collect();
    Ok(Plan { slots })
  }
}

#[derive(Debug)]
enum Slot<'a> {
  Absent,
  Reuse {
    name: &'a str,
    hash: &'a str,
    block: &'a [f32],
  },
  Embed {
    name: &'a str,
    hash: &'a str,
    text: &'a str,
  },
}

/// The embedding work one record needs, and the means to finish it.
#[derive(Debug)]
pub struct Plan<'a> {
  slots: Vec<Slot<'a>>,
}

impl<'a> Plan<'a> {
  /// The texts that must be embedded, in the order [`Self::finish`] expects their vectors.
  pub fn texts(&self) -> Vec<&'a str> {
    self
      .slots
      .iter()
      .filter_map(|slot| match slot {
        Slot::Embed { text, .. } => Some(*text),
        Slot::Absent | Slot::Reuse { .. } => None,
      })
      .collect()
  }

  /// Packs reused and freshly embedded vectors into the record's stored form.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Incompatible`] if `embedded` does not hold exactly one `dimension`-long vector per text of
  /// [`Self::texts`].
  pub fn finish(self, schema: &Schema, dimension: usize, embedded: &[Vec<f32>]) -> Result<RecordVectors> {
    let wanted = self.texts().len();
    if embedded.len() != wanted {
      return Err(Error::Incompatible(format!(
        "{} vectors supplied for {wanted} texts",
        embedded.len()
      )));
    }

    let mut fresh = embedded.iter();
    let mut hashes = BTreeMap::new();
    let mut units: Vec<Option<Vec<f32>>> = Vec::with_capacity(self.slots.len());
    for slot in &self.slots {
      match slot {
        Slot::Absent => units.push(None),
        Slot::Reuse { name, hash, block } => {
          // Stored blocks are scaled by the weights in force when they were written; back to unit length so that
          // `compose` applies today's weights.
          let mut unit = block.to_vec();
          if !normalize(&mut unit) {
            return Err(Error::Incompatible(format!(
              "stored block for `{name}` has no direction"
            )));
          }
          hashes.insert((*name).to_owned(), (*hash).to_owned());
          units.push(Some(unit));
        }
        Slot::Embed { name, hash, .. } => {
          let Some(vector) = fresh.next() else {
            return Err(Error::Incompatible(format!("no vector supplied for `{name}`")));
          };
          hashes.insert((*name).to_owned(), (*hash).to_owned());
          units.push(Some(vector.clone()));
        }
      }
    }

    let composite = schema.compose(dimension, units.iter().map(Option::as_deref))?;
    Ok(RecordVectors { composite, hashes })
  }
}

#[cfg(test)]
pub(crate) mod testing {
  use super::{PreparedRecord, RecordVectors};
  use crate::schema::Schema;

  /// A deterministic stand-in for the model: a unit vector derived from the text's bytes.
  pub(crate) fn fake_embedding(text: &str, dimension: usize) -> Vec<f32> {
    let mut vector = vec![0.0_f32; dimension];
    for (position, byte) in text.bytes().enumerate() {
      vector[(position + usize::from(byte)) % dimension] += f32::from(byte);
    }
    assert!(crate::vector::normalize(&mut vector));
    vector
  }

  /// Embeds a record from scratch with [`fake_embedding`].
  pub(crate) fn fake_vectors(schema: &Schema, dimension: usize, fields: &[(&str, &str)]) -> RecordVectors {
    let prepared = PreparedRecord::new(schema, fields.iter().copied()).unwrap();
    let plan = prepared.plan(schema, dimension, None).unwrap();
    let embedded: Vec<Vec<f32>> = plan
      .texts()
      .iter()
      .map(|text| fake_embedding(text, dimension))
      .collect();
    plan.finish(schema, dimension, &embedded).unwrap()
  }
}

#[cfg(test)]
mod tests {
  use super::testing::{fake_embedding, fake_vectors};
  use super::*;
  use crate::schema::FieldSpec;
  use crate::schema::testing::product_schema;
  use crate::vector::cosine;

  const DIM: usize = 8;

  #[test]
  fn whitespace_is_collapsed_and_blank_text_is_no_text() {
    assert_eq!(
      normalize_text("  Blueberry \n\t Gummies  ").as_deref(),
      Some("Blueberry Gummies")
    );
    assert_eq!(normalize_text(" \n\t "), None);
    assert_eq!(normalize_text(""), None);
  }

  #[test]
  fn text_that_differs_only_in_whitespace_hashes_identically() {
    let a = text_hash(&normalize_text("Blueberry Gummies").unwrap());
    let b = text_hash(&normalize_text(" Blueberry\n Gummies ").unwrap());
    assert_eq!(a, b);
    assert_eq!(a.len(), 32);
    assert_ne!(
      a,
      text_hash(&normalize_text("blueberry gummies").unwrap()),
      "case is a real difference"
    );
  }

  #[test]
  fn unknown_duplicate_and_empty_records_are_refused() {
    let schema = product_schema();
    let unknown = PreparedRecord::new(&schema, [("name", "x"), ("colour", "blue")]).unwrap_err();
    assert!(unknown.to_string().contains("colour"), "{unknown}");
    assert!(PreparedRecord::new(&schema, [("name", "x"), ("name", "y")]).is_err());
    assert!(PreparedRecord::new(&schema, [("name", "  "), ("category", "")]).is_err());
    assert!(PreparedRecord::new(&schema, Vec::<(&str, &str)>::new()).is_err());
  }

  #[test]
  fn a_blank_required_field_is_refused() {
    let schema = Schema::new(vec![
      FieldSpec {
        name: "name".to_owned(),
        weight: 1.0,
        required: true,
        reject_below: None,
      },
      FieldSpec {
        name: "description".to_owned(),
        weight: 1.0,
        required: false,
        reject_below: None,
      },
    ])
    .unwrap();
    assert!(PreparedRecord::new(&schema, [("description", "tasty")]).is_err());
    assert!(PreparedRecord::new(&schema, [("name", " "), ("description", "tasty")]).is_err());
    assert!(PreparedRecord::new(&schema, [("name", "Gummies")]).is_ok());
  }

  #[test]
  fn without_a_baseline_every_field_with_a_value_is_embedded() {
    let schema = product_schema();
    let prepared = PreparedRecord::new(&schema, [("name", "Blueberry Gummies"), ("category", "Edibles")]).unwrap();
    let plan = prepared.plan(&schema, DIM, None).unwrap();
    // Schema order: category, description (absent), name.
    assert_eq!(plan.texts(), ["Edibles", "Blueberry Gummies"]);
  }

  #[test]
  fn only_changed_fields_are_embedded_against_a_baseline() {
    let schema = product_schema();
    let baseline = fake_vectors(
      &schema,
      DIM,
      &[
        ("name", "Blueberry Gummies"),
        ("category", "Edibles"),
        ("description", "Ten vegan gummies."),
      ],
    );
    let revised = PreparedRecord::new(
      &schema,
      [
        ("name", "Blueberry Gummies 100mg"),
        ("category", "Edibles"),
        ("description", " Ten vegan  gummies. "),
      ],
    )
    .unwrap();

    let plan = revised.plan(&schema, DIM, Some(&baseline)).unwrap();
    assert_eq!(
      plan.texts(),
      ["Blueberry Gummies 100mg"],
      "category and (re-spaced) description are reused"
    );

    let vectors = plan
      .finish(&schema, DIM, &[fake_embedding("Blueberry Gummies 100mg", DIM)])
      .unwrap();
    let old = schema.blocks(DIM, &baseline.composite).unwrap();
    let new = schema.blocks(DIM, &vectors.composite).unwrap();
    assert!(
      (cosine(old[0].unwrap(), new[0].unwrap()).unwrap() - 1.0).abs() < 1e-6,
      "category reused"
    );
    assert!(
      (cosine(old[1].unwrap(), new[1].unwrap()).unwrap() - 1.0).abs() < 1e-6,
      "description reused"
    );
    assert!(
      cosine(old[2].unwrap(), new[2].unwrap()).unwrap() < 0.9999,
      "name re-embedded"
    );
    assert_eq!(vectors.hashes.len(), 3);
    assert_ne!(vectors.hashes["name"], baseline.hashes["name"]);
  }

  #[test]
  fn reusing_blocks_gives_the_same_composite_as_embedding_everything() {
    let schema = product_schema();
    let fields = [
      ("name", "Blue Dream Cartridge"),
      ("category", "Vape Pens"),
      ("description", "One gram."),
    ];
    let from_scratch = fake_vectors(&schema, DIM, &fields);

    let prepared = PreparedRecord::new(&schema, fields).unwrap();
    let plan = prepared.plan(&schema, DIM, Some(&from_scratch)).unwrap();
    assert!(plan.texts().is_empty());
    let reused = plan.finish(&schema, DIM, &[]).unwrap();

    assert!((cosine(&from_scratch.composite, &reused.composite).unwrap() - 1.0).abs() < 1e-6);
    assert_eq!(from_scratch.hashes, reused.hashes);
  }

  #[test]
  fn a_field_the_baseline_lacks_is_embedded_and_a_dropped_field_leaves_an_empty_block() {
    let schema = product_schema();
    let baseline = fake_vectors(&schema, DIM, &[("name", "Gummies"), ("description", "Tasty.")]);
    let revised = PreparedRecord::new(&schema, [("name", "Gummies"), ("category", "Edibles")]).unwrap();

    let plan = revised.plan(&schema, DIM, Some(&baseline)).unwrap();
    assert_eq!(plan.texts(), ["Edibles"]);
    let vectors = plan.finish(&schema, DIM, &[fake_embedding("Edibles", DIM)]).unwrap();

    let blocks = schema.blocks(DIM, &vectors.composite).unwrap();
    assert!(blocks[0].is_some() && blocks[1].is_none() && blocks[2].is_some());
    assert!(!vectors.hashes.contains_key("description"));
  }

  #[test]
  fn finish_refuses_the_wrong_number_of_vectors() {
    let schema = product_schema();
    let prepared = PreparedRecord::new(&schema, [("name", "Gummies"), ("category", "Edibles")]).unwrap();
    let one = [fake_embedding("Edibles", DIM)];
    assert!(
      prepared
        .plan(&schema, DIM, None)
        .unwrap()
        .finish(&schema, DIM, &one)
        .is_err()
    );
    assert!(
      prepared
        .plan(&schema, DIM, None)
        .unwrap()
        .finish(&schema, DIM, &[])
        .is_err()
    );
  }

  #[test]
  fn a_baseline_with_a_different_layout_is_refused() {
    let schema = product_schema();
    let foreign = RecordVectors {
      composite: vec![0.5; DIM * 2],
      hashes: BTreeMap::new(),
    };
    let prepared = PreparedRecord::new(&schema, [("name", "Gummies")]).unwrap();
    assert!(prepared.plan(&schema, DIM, Some(&foreign)).is_err());
  }
}
