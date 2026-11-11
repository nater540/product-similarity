//! A small, seedable random number generator (`SplitMix64`).
//!
//! Used where a result has to be reproducible from a seed alone: the trainer orders its examples with it, and the
//! calibration report resamples with it. It is written out here, rather than taken from a crate, so that a seed
//! gives the same sequence on every platform and across dependency upgrades. It is not for anything that needs
//! unpredictability.

/// A generator whose whole sequence is determined by its seed.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
  /// A generator whose whole sequence is determined by `seed`.
  pub fn new(seed: u64) -> Self {
    Self(seed)
  }

  /// The next 64 bits of the sequence.
  pub fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = self.0;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
  }

  /// A number in `0..bound` (zero if `bound` is zero).
  pub fn below(&mut self, bound: usize) -> usize {
    let bound = u64::try_from(bound).unwrap_or(u64::MAX);
    // The modulo bias is below 2^-40 for any list that fits in memory, which is irrelevant for shuffling and
    // for resampling.
    let drawn = self.next_u64().checked_rem(bound).unwrap_or(0);
    usize::try_from(drawn).unwrap_or(0)
  }

  /// Shuffles `items` in place (Fisher-Yates).
  pub fn shuffle<T>(&mut self, items: &mut [T]) {
    for last in (1..items.len()).rev() {
      items.swap(last, self.below(last.saturating_add(1)));
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn a_seed_fixes_the_whole_sequence() {
    let draw = |seed: u64| {
      let mut rng = Rng::new(seed);
      [rng.next_u64(), rng.next_u64(), rng.next_u64()]
    };
    assert_eq!(draw(7), draw(7));
    assert_ne!(draw(7), draw(8));
    // Pinned: a trained model and a reported interval are both reproducible only while this sequence is.
    assert_eq!(
      draw(0),
      [0xE220_A839_7B1D_CDAF, 0x6E78_9E6A_A1B9_65F4, 0x06C4_5D18_8009_454F]
    );
  }

  #[test]
  fn a_bounded_draw_stays_in_range_and_a_shuffle_keeps_every_item() {
    let mut rng = Rng::new(1);
    assert!((0..1000).all(|_| rng.below(7) < 7));
    assert_eq!(rng.below(0), 0);
    let mut items: Vec<usize> = (0..50).collect();
    rng.shuffle(&mut items);
    assert_ne!(items, (0..50).collect::<Vec<_>>());
    items.sort_unstable();
    assert_eq!(items, (0..50).collect::<Vec<_>>());
  }
}
