//! Moving whole owners of a pairs file from one split into another.
//!
//! A model is chosen on pairs it was not trained on, and judged once on pairs that chose nothing. An export
//! comes with two splits, `calibrate` and `holdout`, so choosing anything (a checkpoint, a threshold, a learning
//! rate) had only the holdout to choose with, which uses the holdout up. Carving takes a share of one split's
//! owners and gives them a split of their own, typically `validation` out of `calibrate`.
//!
//! Whole owners move, never single pairs: pairs of one brand are not independent, and a brand on both sides of a
//! split is a leak. Which owners move is decided by a hash of the owner and a salt, so the same command on the
//! same file always moves the same owners, and a later, larger export keeps the ones it already moved.

use anyhow::Context;
use serde_json::{Map, Value};
use sha2::{Digest as _, Sha256};

/// What carving did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Carved {
  /// The file, with the `split` of the moved pairs replaced and every other line as it was.
  pub text: String,
  /// Owners that had pairs in the split carved from.
  pub owners: usize,
  /// Owners that were moved.
  pub owners_moved: usize,
  /// Pairs that were moved.
  pub pairs_moved: usize,
  /// Moved pairs labeled `edit`.
  pub edits_moved: usize,
  /// Moved pairs labeled `recycle`.
  pub recycles_moved: usize,
  /// Pairs of the split carved from that name no owner. They stay: there is no telling whose they are.
  pub without_owner: usize,
}

/// Whether `owner` is among the share `fraction` of owners that move, under `salt`.
///
/// The first eight bytes of the SHA-256 of the salt, a zero byte, and the owner, read as a number, fall below
/// that share of all the numbers there are. A zero byte cannot occur in either text, so no salt and owner can be
/// confused with another salt and owner.
fn moves(owner: &str, salt: &str, fraction: f64) -> bool {
  let mut hasher = Sha256::new();
  hasher.update(salt.as_bytes());
  hasher.update([0_u8]);
  hasher.update(owner.as_bytes());
  let digest = hasher.finalize();
  let mut first = [0_u8; 8];
  for (byte, source) in first.iter_mut().zip(digest.iter()) {
    *byte = *source;
  }
  // 2^64 as a float, and the comparison in 128 bits so that a fraction of 1.0 moves every owner.
  #[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a share of 2^64 between 0 and 1 is a non-negative number that a u128 holds"
  )]
  let below = (fraction.clamp(0.0, 1.0) * 18_446_744_073_709_551_616.0) as u128;
  u128::from(u64::from_be_bytes(first)) < below
}

/// Moves the pairs of a share of the owners in split `from` to split `into`.
///
/// Blank lines, `#` comments, and pairs of every other split pass through byte for byte. A moved line is written
/// back as JSON with its `split` replaced: its keys may come out in another order, its values do not change.
///
/// # Errors
///
/// Returns an error naming the line if a line is not a JSON object, or if `fraction` is not between 0 and 1.
pub fn carve(jsonl: &str, from: &str, into: &str, fraction: f64, salt: &str) -> anyhow::Result<Carved> {
  anyhow::ensure!(
    (0.0..=1.0).contains(&fraction),
    "the fraction is {fraction}; it is a share of the owners, between 0 and 1"
  );
  anyhow::ensure!(from != into, "carving `{from}` into itself would move nothing");
  let mut carved = Carved::default();
  let (mut owners, mut moved) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
  for (index, line) in jsonl.lines().enumerate() {
    let untouched = |carved: &mut Carved| {
      carved.text.push_str(line);
      carved.text.push('\n');
    };
    if line.trim().is_empty() || line.trim_start().starts_with('#') {
      untouched(&mut carved);
      continue;
    }
    let mut pair: Map<String, Value> =
      serde_json::from_str(line).with_context(|| format!("line {} is not a JSON object", index.saturating_add(1)))?;
    if pair.get("split").and_then(Value::as_str) != Some(from) {
      untouched(&mut carved);
      continue;
    }
    let Some(owner) = pair.get("owner").and_then(Value::as_str).map(str::to_owned) else {
      carved.without_owner = carved.without_owner.saturating_add(1);
      untouched(&mut carved);
      continue;
    };
    owners.insert(owner.clone());
    if !moves(&owner, salt, fraction) {
      untouched(&mut carved);
      continue;
    }
    moved.insert(owner);
    carved.pairs_moved = carved.pairs_moved.saturating_add(1);
    match pair.get("label").and_then(Value::as_str) {
      Some("edit") => carved.edits_moved = carved.edits_moved.saturating_add(1),
      Some("recycle") => carved.recycles_moved = carved.recycles_moved.saturating_add(1),
      _ => {}
    }
    pair.insert("split".to_owned(), Value::String(into.to_owned()));
    carved.text.push_str(&Value::Object(pair).to_string());
    carved.text.push('\n');
  }
  carved.owners = owners.len();
  carved.owners_moved = moved.len();
  Ok(carved)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn pair(owner: Option<&str>, split: &str, label: &str) -> String {
    let owner = owner.map_or_else(String::new, |owner| format!(r#""owner": "{owner}", "#));
    format!(r#"{{{owner}"split": "{split}", "label": "{label}", "before": {{"name": "a"}}, "after": {{"name": "b"}}}}"#)
  }

  fn split_of(line: &str) -> String {
    let pair: Map<String, Value> = serde_json::from_str(line).unwrap();
    pair["split"].as_str().unwrap().to_owned()
  }

  /// A file of forty owners with three calibrate pairs and one holdout pair each.
  fn file() -> String {
    let mut lines = vec!["# a comment".to_owned(), String::new()];
    for index in 0..40 {
      let owner = format!("brand-{index}");
      lines.push(pair(Some(&owner), "calibrate", "edit"));
      lines.push(pair(Some(&owner), "calibrate", "recycle"));
      lines.push(pair(Some(&owner), "calibrate", "unsure"));
      lines.push(pair(Some(&owner), "holdout", "edit"));
    }
    lines.push(pair(None, "calibrate", "edit"));
    lines.join("\n")
  }

  #[test]
  fn whole_owners_move_and_every_other_line_is_as_it_was() {
    let file = file();
    let carved = carve(&file, "calibrate", "validation", 0.25, "s1").unwrap();
    assert_eq!(carved.owners, 40);
    assert!(
      (3..=20).contains(&carved.owners_moved),
      "about a quarter of forty: {}",
      carved.owners_moved
    );
    assert_eq!(
      carved.pairs_moved,
      carved.owners_moved * 3,
      "every calibrate pair of a moved owner, and no other"
    );
    assert_eq!(
      (carved.edits_moved, carved.recycles_moved),
      (carved.owners_moved, carved.owners_moved)
    );
    assert_eq!(carved.without_owner, 1);

    let (before, after): (Vec<&str>, Vec<&str>) = (file.lines().collect(), carved.text.lines().collect());
    assert_eq!(before.len(), after.len());
    let mut in_validation = std::collections::BTreeSet::new();
    for (old, new) in before.iter().zip(&after) {
      if old == new {
        continue;
      }
      assert_eq!(
        (split_of(old).as_str(), split_of(new).as_str()),
        ("calibrate", "validation")
      );
      let (old, new): (Map<String, Value>, Map<String, Value>) =
        (serde_json::from_str(old).unwrap(), serde_json::from_str(new).unwrap());
      in_validation.insert(new["owner"].as_str().unwrap().to_owned());
      assert_eq!(
        (&old["before"], &old["label"]),
        (&new["before"], &new["label"]),
        "only the split changed"
      );
    }
    assert_eq!(in_validation.len(), carved.owners_moved);
    // No owner is left on both sides of the new split.
    for line in after.iter().filter(|line| line.starts_with('{')) {
      let pair: Map<String, Value> = serde_json::from_str(line).unwrap();
      if let (Some(owner), Some("calibrate")) = (pair.get("owner").and_then(Value::as_str), pair["split"].as_str()) {
        assert!(!in_validation.contains(owner), "{owner} is in both");
      }
    }
    assert!(
      carved.text.starts_with("# a comment\n\n"),
      "comments and blank lines are kept"
    );
  }

  #[test]
  fn the_same_salt_moves_the_same_owners_and_another_salt_moves_others() {
    let file = file();
    let once = carve(&file, "calibrate", "validation", 0.25, "s1").unwrap();
    assert_eq!(carve(&file, "calibrate", "validation", 0.25, "s1").unwrap(), once);
    assert_ne!(
      carve(&file, "calibrate", "validation", 0.25, "s2").unwrap().text,
      once.text
    );

    // Carving what was already carved moves nothing more: the owners that move have already left.
    let again = carve(&once.text, "calibrate", "validation", 0.25, "s1").unwrap();
    assert_eq!((again.pairs_moved, again.text.as_str()), (0, once.text.as_str()));

    // A larger share with the same salt keeps every owner the smaller share moved.
    let more = carve(&file, "calibrate", "validation", 0.5, "s1").unwrap();
    assert!(more.owners_moved >= once.owners_moved);
    for (small, large) in once.text.lines().zip(more.text.lines()) {
      if small.contains(r#""split":"validation""#) {
        assert_eq!(small, large);
      }
    }
  }

  #[test]
  fn the_whole_and_none_of_a_split_can_be_carved_and_nonsense_is_refused() {
    let file = file();
    assert_eq!(
      carve(&file, "calibrate", "validation", 0.0, "s").unwrap().pairs_moved,
      0
    );
    let all = carve(&file, "calibrate", "validation", 1.0, "s").unwrap();
    assert_eq!((all.owners_moved, all.pairs_moved), (40, 120));
    assert!(carve(&file, "calibrate", "validation", 1.5, "s").is_err());
    assert!(carve(&file, "calibrate", "calibrate", 0.5, "s").is_err());
    let error = carve(
      "{\"split\": \"calibrate\"}\nnot json",
      "calibrate",
      "validation",
      0.5,
      "s",
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("line 2"), "{error:#}");
  }
}
