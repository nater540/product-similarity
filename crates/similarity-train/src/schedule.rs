//! The learning-rate schedule: a linear warm-up to the base rate, then a linear decay to zero.
//!
//! This is the schedule BERT fine-tuning has used since the original paper, and for the same reason: the first
//! updates of a pretrained encoder are computed from a randomly ordered handful of examples and from optimizer
//! statistics that have not settled, so taking them at full size damages weights that were already good.

/// A warm-up-then-decay schedule over a known number of steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearSchedule {
  base: f64,
  warmup: usize,
  total: usize,
}

impl LinearSchedule {
  /// A schedule peaking at `base` after `warmup_ratio` of `total` steps (rounded up).
  pub fn new(base: f64, total: usize, warmup_ratio: f64) -> Self {
    #[expect(
      clippy::cast_precision_loss,
      clippy::cast_possible_truncation,
      clippy::cast_sign_loss,
      reason = "step counts are far below 2^52, and the ratio is clamped to [0, 1] first"
    )]
    let warmup = (total as f64 * warmup_ratio.clamp(0.0, 1.0)).ceil() as usize;
    Self {
      base,
      warmup: warmup.min(total),
      total,
    }
  }

  /// The rate for the zero-based `step`.
  ///
  /// Steps count from one inside the warm-up, so the first step already moves the weights (the widely copied
  /// formulation that counts from zero spends its first batch on a rate of exactly zero). The decay reaches zero
  /// one step after the last, so the last step still moves them too.
  pub fn rate(&self, step: usize) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "step counts are far below 2^52")]
    let fraction = |numerator: usize, denominator: usize| numerator as f64 / denominator.max(1) as f64;
    if step < self.warmup {
      self.base * fraction(step.saturating_add(1), self.warmup)
    } else {
      let decay_steps = self.total.saturating_sub(self.warmup);
      self.base * fraction(self.total.saturating_sub(step), decay_steps)
    }
  }

  /// How many steps the warm-up lasts.
  pub fn warmup_steps(&self) -> usize {
    self.warmup
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-12
  }

  #[test]
  fn the_rate_climbs_to_the_base_then_falls_toward_zero() {
    let schedule = LinearSchedule::new(1.0, 100, 0.1);
    assert_eq!(schedule.warmup_steps(), 10);
    assert!(close(schedule.rate(0), 0.1));
    assert!(close(schedule.rate(4), 0.5));
    assert!(close(schedule.rate(9), 1.0));
    assert!(close(schedule.rate(10), 1.0));
    assert!(close(schedule.rate(55), 0.5));
    assert!(close(schedule.rate(99), 1.0 / 90.0));
  }

  #[test]
  fn every_step_of_a_run_has_a_positive_rate() {
    for total in [1, 2, 3, 7, 50] {
      let schedule = LinearSchedule::new(2e-5, total, 0.1);
      for step in 0..total {
        let rate = schedule.rate(step);
        assert!(rate > 0.0 && rate <= 2e-5, "step {step} of {total}: {rate}");
      }
    }
  }

  #[test]
  fn without_a_warm_up_the_first_step_is_at_the_base_rate() {
    let schedule = LinearSchedule::new(1.0, 10, 0.0);
    assert!(close(schedule.rate(0), 1.0));
    assert!(close(schedule.rate(5), 0.5));
  }

  #[test]
  fn stepping_past_the_end_yields_zero_rather_than_a_negative_rate() {
    let schedule = LinearSchedule::new(1.0, 10, 0.1);
    assert!(close(schedule.rate(10), 0.0));
    assert!(close(schedule.rate(1000), 0.0));
  }
}
