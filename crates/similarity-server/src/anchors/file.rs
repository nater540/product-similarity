//! The file a backup of anchors is kept in.
//!
//! One JSON object per line, like every other file these tools read and write. The first line says what the
//! backup is of: the model and layout the vectors belong to, and when it was taken. Then one line for each
//! record. The last line says how many there were, so that a file that was cut short is known to be: a backup
//! with no last line is not a whole backup, and nothing is restored from one.
//!
//! ```json
//! {"line":"header","format":1,"fingerprint":"f46e48ee0ea2dd02-d256-t256","layout":"...|category,name","dimension":512,"collection":"records","taken_at":1794591798}
//! {"line":"anchor","id":"sku-1","brand_id":"brand-7","anchor":"AACAPwAA...","anchor_quantities":{},"hashes":{"name":"..."},"quantities":{},"revision":3,"updated_at":1794500000}
//! {"line":"end","records":1}
//! ```
//!
//! A vector is written as its bytes (each component a little-endian `f32`) in base64, and not as a list of
//! numbers. The point of a backup is to give back exactly what was stored, which bytes do by construction, and
//! it is half the size.
//!
//! The file holds no text. It holds each record's id and brand, the fingerprints of its text, its quantities,
//! and a vector made from the text it was created with: it deserves the care the collection itself gets.

use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Serialize};
use similarity_core::Quantities;
use similarity_store::ScannedRecord;

use crate::engine::KeptAnchor;

/// The version of the format this code writes, and the only one it reads.
pub const FORMAT: u32 = 1;

/// What a backup is of.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
  /// The version of the format ([`FORMAT`]).
  pub format: u32,
  /// The fingerprint of the model whose vectors these are.
  pub fingerprint: String,
  /// The layout the records were stored under. An anchor means nothing under any other.
  pub layout: String,
  /// How many components an anchor has.
  pub dimension: usize,
  /// The collection the backup was taken of.
  pub collection: String,
  /// When the backup was begun, in seconds since the Unix epoch.
  pub taken_at: u64,
}

/// What a backup keeps of one record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kept {
  /// The record's id.
  pub id: String,
  /// The key of the brand the record is stored under.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub brand_id: Option<String>,
  /// The anchor, as [`encode`] writes a vector.
  pub anchor: String,
  /// The first value the record had for each quantity.
  pub anchor_quantities: Quantities,
  /// The fingerprint of each field's text when the backup was taken.
  pub hashes: BTreeMap<String, String>,
  /// The record's quantities when the backup was taken.
  pub quantities: Quantities,
  /// How many times the record had been written.
  pub revision: u64,
  /// When it was last written, in seconds since the Unix epoch.
  pub updated_at: u64,
}

/// The last line of a backup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct End {
  /// How many records the backup holds.
  pub records: usize,
}

/// One line of a backup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "line", rename_all = "snake_case")]
pub enum Line {
  /// The first line.
  Header(Header),
  /// A record.
  Anchor(Kept),
  /// The last line.
  End(End),
}

/// A vector as a backup writes it: its components as little-endian bytes, in base64.
#[must_use]
pub fn encode(vector: &[f32]) -> String {
  let bytes: Vec<u8> = vector.iter().flat_map(|component| component.to_le_bytes()).collect();
  STANDARD.encode(bytes)
}

/// The vector [`encode`] wrote, which must have `dimension` components.
///
/// # Errors
///
/// Says what is wrong if `text` is not base64, is not `dimension` components long, or holds a component that is
/// not a number: none of which [`encode`] writes for a vector that was stored.
pub fn decode(text: &str, dimension: usize) -> Result<Vec<f32>, String> {
  let bytes = STANDARD
    .decode(text)
    .map_err(|error| format!("the anchor is not base64: {error}"))?;
  let vector: Vec<f32> = bytes
    .chunks_exact(4)
    .filter_map(|chunk| <[u8; 4]>::try_from(chunk).ok())
    .map(f32::from_le_bytes)
    .collect();
  if bytes.len() != dimension.saturating_mul(4) {
    return Err(format!(
      "the anchor is {} bytes, which is not {dimension} components",
      bytes.len()
    ));
  }
  if vector.iter().any(|component| !component.is_finite()) {
    return Err("the anchor holds a component that is not a number".to_owned());
  }
  Ok(vector)
}

impl Kept {
  /// What a backup keeps of a record as a scan that read its anchor listed it. `None` if the scan read no anchor.
  #[must_use]
  pub fn of(record: ScannedRecord) -> Option<Self> {
    Some(Self {
      anchor: encode(record.anchor.as_deref()?),
      id: record.id,
      brand_id: record.brand_id,
      anchor_quantities: record.anchor_quantities,
      hashes: record.hashes,
      quantities: record.quantities,
      revision: record.revision,
      updated_at: record.updated_at,
    })
  }

  /// The anchor that was kept, for a restore.
  ///
  /// # Errors
  ///
  /// As [`decode`].
  pub fn restoring(self, dimension: usize) -> Result<KeptAnchor, String> {
    Ok(KeptAnchor {
      anchor: decode(&self.anchor, dimension)?,
      id: self.id,
      brand_id: self.brand_id,
      anchor_quantities: self.anchor_quantities,
      hashes: self.hashes,
      quantities: self.quantities,
    })
  }
}

/// Writes a backup: the header, a line for each record, and the last line.
#[derive(Debug)]
pub struct Writer<W> {
  out: W,
  records: usize,
}

fn write_line(out: &mut impl Write, line: &Line) -> anyhow::Result<()> {
  serde_json::to_writer(&mut *out, line).context("cannot write the backup")?;
  out.write_all(b"\n").context("cannot write the backup")
}

impl<W: Write> Writer<W> {
  /// Begins a backup with its first line.
  ///
  /// # Errors
  ///
  /// Returns an error if `out` cannot be written to.
  pub fn begin(mut out: W, header: &Header) -> anyhow::Result<Self> {
    write_line(&mut out, &Line::Header(header.clone()))?;
    Ok(Self { out, records: 0 })
  }

  /// Writes one record.
  ///
  /// # Errors
  ///
  /// Returns an error if the line cannot be written.
  pub fn anchor(&mut self, kept: Kept) -> anyhow::Result<()> {
    write_line(&mut self.out, &Line::Anchor(kept))?;
    self.records = self.records.saturating_add(1);
    Ok(())
  }

  /// Ends the backup with its last line, and gives back what it was written to and how many records it holds.
  /// A backup that is not ended is not a backup.
  ///
  /// # Errors
  ///
  /// Returns an error if the last line cannot be written or what was written cannot be flushed.
  pub fn end(mut self) -> anyhow::Result<(W, usize)> {
    write_line(&mut self.out, &Line::End(End { records: self.records }))?;
    self.out.flush().context("cannot write the backup")?;
    Ok((self.out, self.records))
  }
}

/// The records of a backup, one at a time, after its header. It ends in an error, and not quietly, if the file
/// does: the last thing it gives is `Err` unless the last line was there and said the number of records read.
#[derive(Debug)]
pub struct Records<R> {
  lines: std::iter::Enumerate<std::io::Lines<R>>,
  dimension: usize,
  read: usize,
  ended: bool,
}

fn parse(number: usize, line: std::io::Result<String>) -> anyhow::Result<Line> {
  let text = line.with_context(|| format!("line {number}: cannot read the line"))?;
  serde_json::from_str(&text).with_context(|| format!("line {number}: not a line of a backup of anchors"))
}

/// Opens a backup: reads its header and gives its records.
///
/// # Errors
///
/// Returns an error if the first line is not the header of a backup in a format this code reads.
pub fn read<R: BufRead>(reader: R) -> anyhow::Result<(Header, Records<R>)> {
  let mut lines = reader.lines().enumerate();
  let header = match lines.next().map(|(_, line)| parse(1, line)).transpose()? {
    Some(Line::Header(header)) => header,
    Some(_) => anyhow::bail!("line 1: a backup of anchors begins with its header"),
    None => anyhow::bail!("the file is empty: it is not a backup of anchors"),
  };
  anyhow::ensure!(
    header.format == FORMAT,
    "the backup is in format {}, and this version reads format {FORMAT}",
    header.format
  );
  let dimension = header.dimension;
  Ok((
    header,
    Records {
      lines,
      dimension,
      read: 0,
      ended: false,
    },
  ))
}

impl<R: BufRead> Iterator for Records<R> {
  type Item = anyhow::Result<KeptAnchor>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.ended {
      return None;
    }
    let Some((index, line)) = self.lines.next() else {
      // Whatever was read so far, the file does not say it is all there was.
      self.ended = true;
      return Some(Err(anyhow::anyhow!(
        "the file ends after {} record(s) without its last line: it was cut short, and is not a whole backup",
        self.read
      )));
    };
    let number = index.saturating_add(1);
    Some(match parse(number, line) {
      Err(error) => {
        self.ended = true;
        Err(error)
      }
      Ok(Line::Header(_)) => {
        self.ended = true;
        Err(anyhow::anyhow!("line {number}: a second header"))
      }
      Ok(Line::Anchor(kept)) => {
        self.read = self.read.saturating_add(1);
        let id = kept.id.clone();
        kept
          .restoring(self.dimension)
          .map_err(|reason| anyhow::anyhow!("line {number}: record `{id}`: {reason}"))
      }
      Ok(Line::End(end)) => {
        self.ended = true;
        if end.records != self.read {
          Err(anyhow::anyhow!(
            "line {number}: the backup says it holds {} record(s) and {} were read",
            end.records,
            self.read
          ))
        } else if self.lines.next().is_some() {
          Err(anyhow::anyhow!(
            "line {}: something follows the last line",
            number.saturating_add(1)
          ))
        } else {
          return None;
        }
      }
    })
  }
}

/// A backup that was read from its first line to its last and found whole.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Whole {
  /// What it is of.
  pub header: Header,
  /// How many records it holds.
  pub records: usize,
}

/// Reads a whole backup and keeps nothing of it but what it is of and how many records it holds.
///
/// # Errors
///
/// Returns an error, with the line, if the file is not a backup, was cut short, holds a line that cannot be
/// read or an anchor that is not one of its own layout, or holds a record twice.
pub fn check(reader: impl BufRead) -> anyhow::Result<Whole> {
  let (header, records) = read(reader)?;
  let mut ids = HashSet::new();
  for record in records {
    let record = record?;
    anyhow::ensure!(
      ids.insert(record.id.clone()),
      "record `{}` is in the backup twice",
      record.id
    );
  }
  Ok(Whole {
    header,
    records: ids.len(),
  })
}

/// The path a backup is written under until it is whole: beside the one it will have, so that giving it its name
/// stays within one directory.
pub fn partial_path(out: &Path) -> PathBuf {
  let mut name = out.file_name().unwrap_or_default().to_os_string();
  name.push(".partial");
  out.with_file_name(name)
}

/// Creates the file a backup to `out` is written under until it is whole.
///
/// # Errors
///
/// Returns an error if the file cannot be created, saying so plainly if it is already there: another export to
/// the same name is running, or one was stopped part way. Which, only the person running them can tell, so it is
/// never removed here.
pub fn create_partial(out: &Path) -> anyhow::Result<(PathBuf, File)> {
  let partial = partial_path(out);
  match OpenOptions::new().write(true).create_new(true).open(&partial) {
    Ok(file) => Ok((partial, file)),
    Err(error) if error.kind() == ErrorKind::AlreadyExists => anyhow::bail!(
      "{} exists: an export to {} is running, or one was stopped part way. If none is running, remove it",
      partial.display(),
      out.display()
    ),
    Err(error) => Err(error).with_context(|| format!("cannot create {}", partial.display())),
  }
}

/// Gives a whole backup its name, never taking it from a file that has it.
///
/// A rename would replace a file that appeared at `out` while the backup was written, which may be minutes; a
/// link to a name that is taken fails instead. The backup is then left whole under its partial name.
///
/// # Errors
///
/// Returns an error if `out` exists, naming where the backup is, or if the file system cannot link the two names.
pub fn publish(partial: &Path, out: &Path) -> anyhow::Result<()> {
  match std::fs::hard_link(partial, out) {
    Ok(()) => {}
    Err(error) if error.kind() == ErrorKind::AlreadyExists => anyhow::bail!(
      "{} was created while the backup was written, and a backup is not written over another. The backup is \
       whole in {}: give it a name of its own",
      out.display(),
      partial.display()
    ),
    Err(error) => {
      return Err(error).with_context(|| format!("cannot give {} its name, {}", partial.display(), out.display()));
    }
  }
  std::fs::remove_file(partial).with_context(|| {
    format!(
      "the backup is {}, and its partial name {} could not be removed",
      out.display(),
      partial.display()
    )
  })
}

#[cfg(test)]
mod tests {
  use similarity_core::Quantity;

  use super::*;

  #[test]
  fn a_whole_backup_takes_its_name_and_leaves_no_partial_behind() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("anchors.ndjson");
    let (partial, mut file) = create_partial(&out).unwrap();
    file.write_all(b"whole").unwrap();
    drop(file);
    publish(&partial, &out).unwrap();
    assert_eq!(std::fs::read_to_string(&out).unwrap(), "whole");
    assert!(!partial.exists());
  }

  #[test]
  fn a_file_that_appeared_while_the_backup_was_written_is_not_written_over() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("anchors.ndjson");
    let (partial, mut file) = create_partial(&out).unwrap();
    file.write_all(b"whole").unwrap();
    drop(file);
    std::fs::write(&out, "someone else's").unwrap();

    let error = format!("{:#}", publish(&partial, &out).unwrap_err());
    assert_eq!(std::fs::read_to_string(&out).unwrap(), "someone else's");
    assert_eq!(
      std::fs::read_to_string(&partial).unwrap(),
      "whole",
      "and the backup is kept"
    );
    assert!(error.contains(&partial.display().to_string()), "{error}");
  }

  #[test]
  fn an_export_stopped_part_way_is_named_by_the_next_one() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("anchors.ndjson");
    std::fs::write(partial_path(&out), "half").unwrap();
    let error = format!("{:#}", create_partial(&out).unwrap_err());
    assert!(error.contains("stopped part way"), "{error}");
    assert_eq!(std::fs::read_to_string(partial_path(&out)).unwrap(), "half");
  }

  fn header() -> Header {
    Header {
      format: FORMAT,
      fingerprint: "model-a-d2-t128".to_owned(),
      layout: "model-a-d2-t128|category,name".to_owned(),
      dimension: 4,
      collection: "records".to_owned(),
      taken_at: 1_794_591_798,
    }
  }

  fn scanned(id: &str, anchor: [f32; 4]) -> ScannedRecord {
    ScannedRecord {
      id: id.to_owned(),
      brand_id: Some("brand-1".to_owned()),
      hashes: [("name".to_owned(), "00ff".to_owned())].into(),
      quantities: [("weight".to_owned(), Quantity::single(40.0, "g"))].into(),
      anchor_quantities: [("weight".to_owned(), Quantity::single(20.0, "g"))].into(),
      revision: 3,
      updated_at: 1_794_500_000,
      anchor: Some(anchor.to_vec()),
    }
  }

  fn backup(records: &[ScannedRecord]) -> String {
    let mut writer = Writer::begin(Vec::new(), &header()).unwrap();
    for record in records {
      writer.anchor(Kept::of(record.clone()).unwrap()).unwrap();
    }
    let (bytes, written) = writer.end().unwrap();
    assert_eq!(written, records.len());
    String::from_utf8(bytes).unwrap()
  }

  fn failure<T: std::fmt::Debug>(outcome: anyhow::Result<T>) -> String {
    format!("{:#}", outcome.unwrap_err())
  }

  #[test]
  fn a_vector_comes_back_from_its_bytes_as_exactly_the_numbers_it_was() {
    // Values a decimal round trip is least kind to: the smallest, the largest, and neighbors one bit apart.
    let vector = [
      0.0,
      -0.0,
      f32::MIN_POSITIVE,
      f32::MAX,
      f32::MIN,
      0.1,
      f32::from_bits(0.1_f32.to_bits() + 1),
      1.0 - f32::EPSILON,
      0.365_148_37,
    ];
    let decoded = decode(&encode(&vector), vector.len()).unwrap();
    let bits = |vector: &[f32]| vector.iter().map(|component| component.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&decoded), bits(&vector));
    assert_eq!(encode(&[1.0]), "AACAPw==", "little-endian bytes, in standard base64");
    assert_eq!(decode("", 0).unwrap(), Vec::<f32>::new());
  }

  #[test]
  fn an_anchor_that_is_not_a_vector_of_the_layout_is_refused_with_the_reason() {
    assert!(decode("not base64!", 1).unwrap_err().contains("not base64"));
    let two = encode(&[0.6, 0.8]);
    assert!(
      decode(&two, 3)
        .unwrap_err()
        .contains("8 bytes, which is not 3 components")
    );
    assert!(decode(&two, 1).unwrap_err().contains("not 1 components"));
    // Seven bytes are not two components either, and are not read as one.
    let ragged = STANDARD.encode([0_u8; 7]);
    assert!(decode(&ragged, 2).unwrap_err().contains("7 bytes"));
    for not_a_number in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
      let reason = decode(&encode(&[0.5, not_a_number]), 2).unwrap_err();
      assert!(reason.contains("not a number"), "{reason}");
    }
  }

  #[test]
  fn a_backup_is_a_header_a_line_for_each_record_and_a_last_line_and_reads_back_as_what_was_kept() {
    let records = [
      scanned("sku-1", [1.0, 0.0, 0.0, 0.0]),
      scanned("sku-2", [0.0, 0.6, 0.8, 0.0]),
    ];
    let text = backup(&records);
    let lines: Vec<serde_json::Value> = text.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    let kinds: Vec<&str> = lines.iter().map(|line| line["line"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["header", "anchor", "anchor", "end"]);
    assert_eq!(
      (
        lines[0]["format"].as_u64(),
        lines[0]["dimension"].as_u64(),
        lines[3]["records"].as_u64()
      ),
      (Some(1), Some(4), Some(2))
    );
    assert_eq!(
      lines[1]["anchor"].as_str(),
      Some(encode(&[1.0, 0.0, 0.0, 0.0]).as_str())
    );

    let (read_header, read_records) = read(text.as_bytes()).unwrap();
    assert_eq!(read_header, header());
    let read_records: Vec<KeptAnchor> = read_records.map(Result::unwrap).collect();
    assert_eq!(read_records.len(), 2);
    let second = &read_records[1];
    assert_eq!(
      (second.id.as_str(), second.brand_id.as_deref()),
      ("sku-2", Some("brand-1"))
    );
    assert_eq!(second.anchor, [0.0, 0.6, 0.8, 0.0]);
    assert_eq!(second.anchor_quantities["weight"], Quantity::single(20.0, "g"));
    assert_eq!(second.quantities["weight"], Quantity::single(40.0, "g"));
    assert_eq!(second.hashes["name"], "00ff");

    assert_eq!(
      check(text.as_bytes()).unwrap(),
      Whole {
        header: header(),
        records: 2
      }
    );
    // A backup of nothing is a whole backup.
    assert_eq!(check(backup(&[]).as_bytes()).unwrap().records, 0);
    // A record a scan read no anchor of is not something to keep.
    let unread = ScannedRecord {
      anchor: None,
      ..scanned("sku-3", [1.0, 0.0, 0.0, 0.0])
    };
    assert_eq!(Kept::of(unread), None);
  }

  #[test]
  fn a_backup_that_was_cut_short_is_not_a_whole_backup() {
    let text = backup(&[
      scanned("sku-1", [1.0, 0.0, 0.0, 0.0]),
      scanned("sku-2", [0.0, 1.0, 0.0, 0.0]),
    ]);
    let lines: Vec<&str> = text.lines().collect();

    // Without its last line: every record that is there can be read, and then the reading fails.
    let cut = lines[..3].join("\n");
    let (_, mut records) = read(cut.as_bytes()).unwrap();
    assert!(records.next().unwrap().is_ok() && records.next().unwrap().is_ok());
    let end = failure(records.next().unwrap());
    assert!(end.contains("after 2 record(s) without its last line"), "{end}");
    assert!(records.next().is_none(), "and it fails once");
    assert!(failure(check(cut.as_bytes())).contains("cut short"));

    // With a record missing from the middle, the last line gives it away.
    let short = [lines[0], lines[1], lines[3]].join("\n");
    let missing = failure(check(short.as_bytes()));
    assert!(
      missing.contains("line 3: the backup says it holds 2 record(s) and 1 were read"),
      "{missing}"
    );

    // Cut in the middle of a line.
    let torn = &text[..text.len() - lines[3].len() - lines[2].len() / 2];
    assert!(failure(check(torn.as_bytes())).contains("line 3: not a line of a backup"));
  }

  #[test]
  fn a_file_that_is_not_a_backup_is_refused_at_the_line_that_says_so() {
    let text = backup(&[scanned("sku-1", [1.0, 0.0, 0.0, 0.0])]);
    let lines: Vec<&str> = text.lines().collect();

    assert!(failure(read("".as_bytes())).contains("the file is empty"));
    assert!(failure(read(lines[1].as_bytes())).contains("line 1: a backup of anchors begins with its header"));
    assert!(
      failure(read(
        r#"{"id": "sku-1", "record": {"name": "an export, not a backup"}}"#.as_bytes()
      ))
      .contains("line 1")
    );
    let newer = lines[0].replace(r#""format":1"#, r#""format":2"#);
    assert!(failure(read(newer.as_bytes())).contains("format 2, and this version reads format 1"));

    let again = [lines[0], lines[1], lines[0], lines[2]].join("\n");
    assert!(failure(check(again.as_bytes())).contains("line 3: a second header"));
    let after = [lines[0], lines[1], lines[2], lines[1]].join("\n");
    assert!(failure(check(after.as_bytes())).contains("line 4: something follows the last line"));
    let twice = [lines[0], lines[1], lines[1], r#"{"line":"end","records":2}"#].join("\n");
    assert!(failure(check(twice.as_bytes())).contains("record `sku-1` is in the backup twice"));

    // An anchor of another width than the header says, and a key no line has.
    let narrow = lines[1].replace(&encode(&[1.0, 0.0, 0.0, 0.0]), &encode(&[1.0, 0.0]));
    let wrong = failure(check([lines[0], &narrow, lines[2]].join("\n").as_bytes()));
    assert!(
      wrong.contains("line 2: record `sku-1`: the anchor is 8 bytes"),
      "{wrong}"
    );
    let stray = lines[1].replace(r#""id":"#, r#""text":"blueberry","id":"#);
    assert!(failure(check([lines[0], &stray, lines[2]].join("\n").as_bytes())).contains("line 2: not a line"));
  }
}
