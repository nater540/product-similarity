//! Rules of the run registry that must hold for every input.

use proptest::prelude::*;

use super::{slug, utc_stamp};

proptest! {
  /// A run's id is a directory name made partly from text a caller chose (an output directory, a file name). It
  /// must never be able to name a place outside the runs root, whatever that text is and whenever the run starts.
  #[test]
  fn an_id_is_always_one_safe_path_component(text in "\\PC{0,80}", seconds in 0_u64..=253_402_300_799) {
    let id = format!("{}-train-{}", utc_stamp(seconds), slug(&text));
    prop_assert!(id.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'), "{}", id);
    prop_assert!(!id.starts_with('-') && !id.ends_with('-') && !id.contains("--"), "{}", id);
    prop_assert_eq!(std::path::Path::new(&id).components().count(), 1);
    prop_assert!(id.len() <= 16 + 7 + 40, "{}", id);
    // Sixteen characters of time, so that names sort in order of time.
    let stamp = utc_stamp(seconds);
    prop_assert_eq!(stamp.len(), 16);
    prop_assert!(stamp.ends_with('Z') && stamp.as_bytes().get(8) == Some(&b'T'), "{}", stamp);
  }

  /// A later second is never an earlier name.
  #[test]
  fn ids_sort_in_the_order_the_runs_started(a in 0_u64..=253_402_300_799, b in 0_u64..=253_402_300_799) {
    prop_assert_eq!(a.cmp(&b), utc_stamp(a).cmp(&utc_stamp(b)));
  }
}
