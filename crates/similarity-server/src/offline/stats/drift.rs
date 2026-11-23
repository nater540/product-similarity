//! What replayed histories say about drift: every number in the drift report and its events.
//!
//! Three questions, each answered from what the replay measured:
//!
//! - **Where does `anchor_threshold` belong?** Below how far any legitimate history drifts, and above how far the
//!   recycles drift that nothing else catches. It is not placed in a gap between two labels, as the overall
//!   threshold is: most recycles are caught by the pair rules at one step, and say nothing about drift.
//! - **What does the configured policy do to these histories?** How many legitimate ones it blocks, and by which
//!   rule; how many recycles it catches, and when.
//! - **Do quantities need a drift check?** A factor limit holds each version to the one before it. Whether any
//!   recycle walks a quantity far in steps that each stay small is a count here, not an opinion.
//!
//! A history is in two parts. An `edit` history is legitimate throughout. A `recycle` with `recycled_at` is
//! legitimate before that version and recycled from it on. A `recycle` without one is recycled somewhere, so
//! none of it can be called legitimate. A truncated history (one whose first version is not its creation) is
//! counted in what the policy does and left out of where the threshold belongs: its drift is understated.

use std::collections::BTreeMap;

use similarity_core::{Policy, Reason};

use super::{Interval, Rate, bootstrap, number};
use crate::offline::drift::{DriftRun, Replayed, Step};
use crate::offline::pairs::Label;

/// When a recycled history was first rejected, against when it was recycled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Caught {
  /// At the version it was recycled at.
  AtRecycle,
  /// After it: recycled versions were accepted first.
  Later,
  /// Before it: a legitimate version was rejected.
  Early,
  /// At some version, for a history that does not say where it was recycled.
  Somewhere,
  /// Never.
  Never,
}

impl Caught {
  /// The word for it, as the events write it.
  pub fn as_str(self) -> &'static str {
    match self {
      Self::AtRecycle => "at_recycle",
      Self::Later => "later",
      Self::Early => "early",
      Self::Somewhere => "somewhere",
      Self::Never => "never",
    }
  }
}

/// How far one quantity went over one history.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryQuantity {
  /// The quantity.
  pub quantity: String,
  /// The largest factor between one version and the next.
  pub max_step_factor: Option<f64>,
  /// The largest factor between a version and the first value the record had.
  pub max_first_factor: Option<f64>,
  /// Whether it was ever in a unit that cannot be compared with its first.
  pub first_incomparable: bool,
  /// Whether it ever moved between zero and a value above zero, against its first.
  pub first_zero_crossing: bool,
}

/// One history, summarized.
#[derive(Debug, Clone, PartialEq)]
pub struct HistorySummary {
  /// How many steps it has.
  pub steps: usize,
  /// The lowest similarity to the anchor it reaches, and the step it reaches it at.
  pub min_anchor: Option<(f32, usize)>,
  /// The similarity to the anchor at its last version.
  pub final_anchor: Option<f32>,
  /// The first step the configured policy rejects.
  pub first_rejected: Option<usize>,
  /// The first step the pair rules reject: the policy without the drift check.
  pub first_rejected_by_pair_rules: Option<usize>,
  /// The first step whose similarity to the anchor is below the drift threshold, if one is configured.
  pub first_anchor_drift: Option<usize>,
  /// For a recycle, when it was caught.
  pub caught: Option<Caught>,
  /// Each quantity the history has.
  pub quantities: Vec<HistoryQuantity>,
}

fn lowest_anchor<'a>(steps: impl IntoIterator<Item = &'a Step>) -> Option<(f32, usize)> {
  steps
    .into_iter()
    .filter_map(|step| Some((step.evaluation.anchor_similarity?, step.step)))
    .min_by(|a, b| a.0.total_cmp(&b.0))
}

/// The steps of a history that are known to be legitimate.
fn legitimate(replayed: &Replayed) -> &[Step] {
  let history = &replayed.history;
  match (history.label, history.recycled_at) {
    (Some(Label::Edit), _) => &replayed.steps,
    // Steps are numbered from 1, so the steps before version `at` are the first `at - 1` of them.
    (Some(Label::Recycle), Some(at)) => replayed.steps.get(..at.saturating_sub(1)).unwrap_or(&[]),
    _ => &[],
  }
}

/// The steps of a history from the version it was recycled at. All of them, when it does not say which that was.
fn recycled(replayed: &Replayed) -> &[Step] {
  let history = &replayed.history;
  match (history.label, history.recycled_at) {
    (Some(Label::Recycle), Some(at)) => replayed.steps.get(at.saturating_sub(1)..).unwrap_or(&[]),
    (Some(Label::Recycle), None) => &replayed.steps,
    _ => &[],
  }
}

fn largest(factors: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
  factors.into_iter().flatten().max_by(f64::total_cmp)
}

fn quantities_of(steps: &[Step]) -> Vec<HistoryQuantity> {
  let mut names: Vec<&str> = steps
    .iter()
    .flat_map(|step| step.evaluation.quantities.iter().chain(&step.anchor_quantities))
    .map(|score| score.field.as_str())
    .collect();
  names.sort_unstable();
  names.dedup();
  names
    .into_iter()
    .map(|name| {
      let of = |scores: fn(&Step) -> &[similarity_core::QuantityScore]| {
        steps
          .iter()
          .flat_map(move |step| scores(step).iter().filter(move |score| score.field == name))
      };
      let against_first = || of(|step| &step.anchor_quantities);
      HistoryQuantity {
        quantity: name.to_owned(),
        max_step_factor: largest(of(|step| &step.evaluation.quantities).map(|score| score.factor)),
        max_first_factor: largest(against_first().map(|score| score.factor)),
        first_incomparable: against_first().any(|score| score.incomparable),
        first_zero_crossing: against_first().any(|score| score.zero_crossing),
      }
    })
    .collect()
}

/// Summarizes one replayed history under `policy`.
pub fn summarize(replayed: &Replayed, policy: &Policy) -> HistorySummary {
  let steps = &replayed.steps;
  let first = |rejected: &dyn Fn(&Step) -> bool| steps.iter().find(|step| rejected(step)).map(|step| step.step);
  let first_rejected = first(&|step| !step.evaluation.accepted());
  let caught =
    (replayed.history.label == Some(Label::Recycle)).then_some(match (first_rejected, replayed.history.recycled_at) {
      (None, _) => Caught::Never,
      (Some(_), None) => Caught::Somewhere,
      (Some(step), Some(at)) if step < at => Caught::Early,
      (Some(step), Some(at)) if step == at => Caught::AtRecycle,
      (Some(_), Some(_)) => Caught::Later,
    });
  HistorySummary {
    steps: steps.len(),
    min_anchor: lowest_anchor(steps),
    final_anchor: steps.last().and_then(|step| step.evaluation.anchor_similarity),
    first_rejected,
    first_rejected_by_pair_rules: first(&|step| !step.accepted_by_pair_rules),
    first_anchor_drift: policy
      .anchor_threshold
      .and_then(|threshold| first(&|step| step.drifted_below(threshold))),
    caught,
    quantities: quantities_of(steps),
  }
}

/// How many histories and steps a run has, by label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Totals {
  /// Histories.
  pub histories: usize,
  /// Labeled `edit`.
  pub edits: usize,
  /// Labeled `recycle`.
  pub recycles: usize,
  /// With any other label: replayed and shown, counted nowhere else.
  pub others: usize,
  /// Whose first version is not their creation, with no original given.
  pub truncated: usize,
  /// Steps in all.
  pub steps: usize,
}

/// Counts the histories of `run`.
pub fn totals(run: &DriftRun) -> Totals {
  let with = |label: Option<Label>| {
    run
      .histories
      .iter()
      .filter(|replayed| replayed.history.label == label)
      .count()
  };
  Totals {
    histories: run.histories.len(),
    edits: with(Some(Label::Edit)),
    recycles: with(Some(Label::Recycle)),
    others: with(None),
    truncated: run
      .histories
      .iter()
      .filter(|replayed| replayed.history.truncated)
      .count(),
    steps: run.histories.iter().map(|replayed| replayed.steps.len()).sum(),
  }
}

/// How low the similarity to the anchor goes, over the histories of one kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftRange {
  /// How many histories.
  pub count: usize,
  /// The lowest any of them reaches.
  pub lowest: f64,
  /// The fifth percentile, by nearest rank: with fewer than twenty histories, the lowest.
  pub p5: f64,
  /// The median.
  pub median: f64,
  /// The highest: the history that drifted least.
  pub highest: f64,
}

/// The lowest similarity to the anchor each history reaches, over its legitimate part (`Label::Edit`) or its
/// recycled part (`Label::Recycle`). Truncated histories are left out.
pub fn drift_range(run: &DriftRun, part: Label) -> Option<DriftRange> {
  let mut lows: Vec<f64> = run
    .histories
    .iter()
    .filter(|replayed| !replayed.history.truncated)
    .filter_map(|replayed| {
      let steps = match part {
        Label::Edit => legitimate(replayed),
        Label::Recycle => recycled(replayed),
      };
      lowest_anchor(steps).map(|(similarity, _)| f64::from(similarity))
    })
    .collect();
  lows.sort_by(f64::total_cmp);
  let rank = |percent: usize| {
    let rank = lows.len().saturating_mul(percent).div_ceil(100).max(1);
    lows.get(rank.saturating_sub(1)).copied()
  };
  Some(DriftRange {
    count: lows.len(),
    lowest: *lows.first()?,
    p5: rank(5)?,
    median: rank(50)?,
    highest: *lows.last()?,
  })
}

/// A share of some histories, with the interval that resampling their brands gives.
fn rate(histories: &[&Replayed], is_error: impl Fn(&Replayed) -> bool, seed: u64) -> Rate {
  let errors = histories.iter().filter(|replayed| is_error(replayed)).count();
  let mut clusters: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
  let mut every_one_owned = true;
  for replayed in histories {
    let Some(brand_id) = replayed.history.brand_id.as_deref() else {
      every_one_owned = false;
      continue;
    };
    let cluster = clusters.entry(brand_id).or_default();
    cluster.0 = cluster.0.saturating_add(usize::from(is_error(replayed)));
    cluster.1 = cluster.1.saturating_add(1);
  }
  // Histories of one brand are not independent, so the brands are what is resampled (D17). Without a brand on
  // every history, or with only one brand, there is nothing to resample.
  let interval: Option<Interval> = (every_one_owned && clusters.len() > 1)
    .then(|| bootstrap(&clusters.into_values().collect::<Vec<_>>(), seed))
    .flatten();
  Rate {
    errors,
    of: histories.len(),
    rate: (!histories.is_empty()).then(|| number(errors) / number(histories.len())),
    interval,
  }
}

/// What the configured policy does to legitimate histories.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blocked {
  /// Rejected at some step by the policy as configured.
  pub by_policy: Rate,
  /// Rejected at some step by the pair rules: the policy without the drift check.
  pub by_pair_rules: Rate,
  /// Rejected by the drift check at a step the pair rules pass, and by the pair rules at none.
  pub by_drift_alone: Rate,
}

/// How many `edit` histories the configured policy blocks, and by which rule.
pub fn blocked(run: &DriftRun, seed: u64) -> Blocked {
  let edits: Vec<&Replayed> = run
    .histories
    .iter()
    .filter(|replayed| replayed.history.label == Some(Label::Edit))
    .collect();
  let any = |replayed: &Replayed, rejected: fn(&Step) -> bool| replayed.steps.iter().any(rejected);
  let by_policy = |replayed: &Replayed| any(replayed, |step| !step.evaluation.accepted());
  let by_pair_rules = |replayed: &Replayed| any(replayed, |step| !step.accepted_by_pair_rules);
  Blocked {
    by_policy: rate(&edits, by_policy, seed),
    by_pair_rules: rate(&edits, by_pair_rules, seed),
    by_drift_alone: rate(&edits, |replayed| by_policy(replayed) && !by_pair_rules(replayed), seed),
  }
}

/// What the configured policy does to recycled histories.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Recycles {
  /// First rejected at the version they were recycled at.
  pub at_recycle: usize,
  /// First rejected later than that.
  pub later: usize,
  /// First rejected before it: a legitimate version was the one refused.
  pub early: usize,
  /// Rejected somewhere, for histories that do not say where they were recycled.
  pub somewhere: usize,
  /// Never rejected.
  pub never: usize,
  /// Rejected, and only ever by the drift check: the recycles the anchor exists for.
  pub by_drift_alone: usize,
}

/// When the configured policy catches each `recycle` history.
pub fn recycles(run: &DriftRun, policy: &Policy) -> Recycles {
  let mut counts = Recycles::default();
  for replayed in &run.histories {
    let slot = match summarize(replayed, policy).caught {
      Some(Caught::AtRecycle) => &mut counts.at_recycle,
      Some(Caught::Later) => &mut counts.later,
      Some(Caught::Early) => &mut counts.early,
      Some(Caught::Somewhere) => &mut counts.somewhere,
      Some(Caught::Never) => &mut counts.never,
      None => continue,
    };
    *slot = slot.saturating_add(1);
    let rejected = replayed.steps.iter().any(|step| !step.evaluation.accepted());
    let by_pair_rules = replayed.steps.iter().any(|step| !step.accepted_by_pair_rules);
    if rejected && !by_pair_rules {
      counts.by_drift_alone = counts.by_drift_alone.saturating_add(1);
    }
  }
  counts
}

/// Where the drift threshold could go, read from the histories.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchorSuggestion {
  /// The lowest similarity to the anchor any legitimate history reaches. A drift threshold has to be below it,
  /// or it blocks a product that only ever was itself.
  pub legit_low: Option<f64>,
  /// Recycled histories the pair rules never reject once recycled: the only ones a drift check adds anything
  /// for.
  pub uncaught: usize,
  /// The highest similarity to the anchor, below `legit_low`, that one of those sinks to.
  pub next_below: Option<f64>,
  /// Halfway between the two: below every legitimate history, with the same room on both sides. `None` when no
  /// uncaught recycle sinks below every legitimate history, in which case no drift threshold separates them.
  pub anchor_threshold: Option<f64>,
  /// Half the gap.
  pub margin: Option<f64>,
  /// How many of the uncaught recycles that threshold rejects.
  pub recycles_caught: usize,
  /// The suggestion is above the overall threshold. At a record's first edit the anchor is the version before,
  /// so such a drift threshold is stricter than the overall one on every first edit.
  pub above_threshold: bool,
}

/// Places the drift threshold between how far legitimate histories drift and how far the recycles drift that the
/// pair rules miss. Truncated histories are left out: they drifted further than they show.
pub fn anchor_suggestion(run: &DriftRun, policy: &Policy) -> AnchorSuggestion {
  let measured = || run.histories.iter().filter(|replayed| !replayed.history.truncated);
  let low = |steps: &[Step]| lowest_anchor(steps).map(|(similarity, _)| f64::from(similarity));
  let legit_low = measured()
    .filter_map(|replayed| low(legitimate(replayed)))
    .min_by(f64::total_cmp);
  let uncaught: Vec<f64> = measured()
    .filter(|replayed| {
      let part = recycled(replayed);
      !part.is_empty() && part.iter().all(|step| step.accepted_by_pair_rules)
    })
    .filter_map(|replayed| low(recycled(replayed)))
    .collect();
  // The uncaught recycles that sink below every legitimate history: the ones a threshold can separate. One that
  // only reaches as low as a legitimate history does cannot be told from it by any threshold.
  let below: Vec<f64> = legit_low.map_or_else(Vec::new, |floor| {
    uncaught.iter().copied().filter(|lowest| *lowest < floor).collect()
  });
  let next_below = below.iter().copied().max_by(f64::total_cmp);
  let anchor_threshold = legit_low
    .zip(next_below)
    .map(|(floor, next)| f64::midpoint(floor, next));
  AnchorSuggestion {
    legit_low,
    uncaught: uncaught.len(),
    next_below,
    anchor_threshold,
    margin: legit_low.zip(next_below).map(|(floor, next)| (floor - next) / 2.0),
    // The threshold lies above the nearest of them, so it rejects them all.
    recycles_caught: anchor_threshold.map_or(0, |_| below.len()),
    above_threshold: anchor_threshold.is_some_and(|threshold| threshold > f64::from(policy.threshold)),
  }
}

/// How far one quantity goes across the histories, per step and against its first value.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantityDrift {
  /// The quantity.
  pub quantity: String,
  /// Among legitimate histories: the largest factor between two versions, and against the first value.
  pub edits: (Option<f64>, Option<f64>),
  /// Among recycled histories: the same two.
  pub recycles: (Option<f64>, Option<f64>),
  /// Recycled histories that end up further from their first value than any legitimate history does, in steps
  /// none of which is larger than a legitimate history's largest. A limit on each step, placed from the edits,
  /// cannot catch these; a limit against the first value could. Zero, on enough histories, is the evidence that
  /// quantities need no drift check.
  pub walked: usize,
  /// Histories in which it became incomparable with its first unit: legitimate, then recycled.
  pub first_incomparable: (usize, usize),
  /// Histories in which it crossed zero against its first value: legitimate, then recycled.
  pub first_zero_crossings: (usize, usize),
}

/// The quantity drift of every quantity some history has. Histories with another label are left out.
pub fn quantity_drift(run: &DriftRun) -> Vec<QuantityDrift> {
  let of = |label: Label| -> Vec<Vec<HistoryQuantity>> {
    run
      .histories
      .iter()
      .filter(|replayed| replayed.history.label == Some(label))
      .map(|replayed| quantities_of(&replayed.steps))
      .collect()
  };
  let (edits, recycles) = (of(Label::Edit), of(Label::Recycle));
  let mut names: Vec<&str> = edits
    .iter()
    .chain(&recycles)
    .flatten()
    .map(|quantity| quantity.quantity.as_str())
    .collect();
  names.sort_unstable();
  names.dedup();
  names
    .into_iter()
    .map(|name| {
      let of_this = |histories: &'_ [Vec<HistoryQuantity>]| -> Vec<HistoryQuantity> {
        histories
          .iter()
          .filter_map(|quantities| quantities.iter().find(|quantity| quantity.quantity == name).cloned())
          .collect()
      };
      let (edits, recycles) = (of_this(&edits), of_this(&recycles));
      let extremes = |histories: &[HistoryQuantity]| {
        (
          largest(histories.iter().map(|quantity| quantity.max_step_factor)),
          largest(histories.iter().map(|quantity| quantity.max_first_factor)),
        )
      };
      let flags = |pick: fn(&HistoryQuantity) -> bool| {
        (
          edits.iter().filter(|quantity| pick(quantity)).count(),
          recycles.iter().filter(|quantity| pick(quantity)).count(),
        )
      };
      let (edit_step, edit_first) = extremes(&edits);
      // A quantity no legitimate history ever changes has a largest factor of one: unchanged.
      let (step_ceiling, first_ceiling) = (edit_step.unwrap_or(1.0), edit_first.unwrap_or(1.0));
      QuantityDrift {
        quantity: name.to_owned(),
        edits: (edit_step, edit_first),
        recycles: extremes(&recycles),
        walked: recycles
          .iter()
          .filter(|quantity| {
            quantity.max_first_factor.is_some_and(|factor| factor > first_ceiling)
              && quantity.max_step_factor.is_none_or(|factor| factor <= step_ceiling)
          })
          .count(),
        first_incomparable: flags(|quantity| quantity.first_incomparable),
        first_zero_crossings: flags(|quantity| quantity.first_zero_crossing),
      }
    })
    .collect()
}

/// The first step at which the configured policy rejects `replayed`, with its reasons: what the report lists a
/// blocked or missed history by.
pub fn first_rejection(replayed: &Replayed) -> Option<(&Step, &[Reason])> {
  let step = replayed.steps.iter().find(|step| !step.evaluation.accepted())?;
  Some((step, &step.evaluation.reasons))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::offline::drift::testing::{line, policy, replayed};

  type Versions<'a> = Vec<(&'a str, Option<&'a str>, Option<f64>)>;

  /// A walk from gummies to a cartridge, a word at a time. Legitimate for one step, recycled from the second.
  fn walk() -> Versions<'static> {
    vec![
      ("blueberry gummies edibles sweet", None, Some(40.0)),
      ("blueberry gummies edibles sweet vegan", None, Some(40.0)),
      ("blueberry gummies edibles cartridge vegan", None, Some(40.0)),
      ("blueberry vape edibles cartridge vegan", None, Some(40.0)),
      ("sativa vape resin cartridge vegan", None, Some(40.0)),
      ("sativa vape resin cartridge distillate", None, Some(40.0)),
    ]
  }

  /// A product that stays itself: a word added, a word changed back.
  fn steady(name: &'static str) -> Versions<'static> {
    vec![
      (name, None, Some(3.5)),
      ("indica flower premium", None, Some(3.5)),
      (name, None, Some(3.6)),
    ]
  }

  fn run() -> DriftRun {
    replayed(&[
      line(
        "steady-1",
        r#""brand_id": "b1", "label": "edit", "#,
        &steady("indica flower"),
      ),
      line(
        "steady-2",
        r#""brand_id": "b2", "label": "edit", "#,
        &steady("indica flower small"),
      ),
      line(
        "walk",
        r#""brand_id": "b3", "label": "recycle", "recycled_at": 2, "#,
        &walk(),
      ),
      line(
        "jump",
        r#""brand_id": "b3", "label": "recycle", "recycled_at": 1, "#,
        &[
          ("blueberry gummies", None, Some(40.0)),
          ("sativa vape cartridge", None, Some(40.0)),
        ],
      ),
      line("other", r#""label": "unsure", "#, &steady("sativa flower")),
    ])
  }

  #[test]
  fn a_history_is_summarized_by_how_far_it_went_and_where_it_was_stopped() {
    let run = run();
    let policy = policy();
    let walk = summarize(&run.histories[2], &policy);
    assert_eq!(walk.steps, 5);
    let (lowest, at) = walk.min_anchor.unwrap();
    // The fourth version already shares no word with the first, and the earliest step at the lowest is the one named.
    assert!(lowest < 0.3 && at == 4, "{lowest} at {at}");
    assert_eq!(walk.final_anchor, Some(lowest));
    assert_eq!(walk.first_rejected_by_pair_rules, None, "no step is large");
    assert_eq!(
      walk.first_rejected, walk.first_anchor_drift,
      "the drift check is what stops it"
    );
    assert!(walk.first_rejected.unwrap() > 2);
    assert_eq!(
      walk.caught,
      Some(Caught::Later),
      "recycled at version 2, stopped after it"
    );

    let jump = summarize(&run.histories[3], &policy);
    assert_eq!(
      (jump.first_rejected, jump.first_rejected_by_pair_rules),
      (Some(1), Some(1))
    );
    assert_eq!(jump.caught, Some(Caught::AtRecycle));

    let steady = summarize(&run.histories[0], &policy);
    assert_eq!((steady.first_rejected, steady.caught), (None, None));
    assert!((steady.quantities[0].max_step_factor.unwrap() - 3.6 / 3.5).abs() < 1e-9);
    assert!((steady.quantities[0].max_first_factor.unwrap() - 3.6 / 3.5).abs() < 1e-9);
  }

  #[test]
  fn the_histories_are_counted_by_label() {
    let counted = totals(&run());
    assert_eq!(
      (counted.histories, counted.edits, counted.recycles, counted.others),
      (5, 2, 2, 1)
    );
    assert_eq!((counted.steps, counted.truncated), (2 + 2 + 5 + 1 + 2, 0));
  }

  #[test]
  fn drift_is_ranged_over_the_legitimate_part_and_the_recycled_part_apart() {
    let run = run();
    // Two edit histories, and the one legitimate step of the walk.
    let legitimate = drift_range(&run, Label::Edit).unwrap();
    assert_eq!(legitimate.count, 3);
    assert!(
      legitimate.lowest > 0.6 && legitimate.lowest <= legitimate.median,
      "{legitimate:?}"
    );
    assert!(
      (legitimate.p5 - legitimate.lowest).abs() < 1e-12,
      "with three histories the fifth percentile is the lowest"
    );
    let recycled = drift_range(&run, Label::Recycle).unwrap();
    assert_eq!(recycled.count, 2);
    assert!(
      recycled.lowest < 0.3 && recycled.highest < legitimate.lowest,
      "{recycled:?}"
    );
    assert_eq!(drift_range(&replayed(&[]), Label::Edit), None);
  }

  #[test]
  fn the_drift_threshold_goes_between_legitimate_drift_and_the_recycles_nothing_else_catches() {
    let run = run();
    let suggestion = anchor_suggestion(&run, &policy());
    let legit_low = suggestion.legit_low.unwrap();
    // Only the walk is uncaught by the pair rules. The jump is caught at its one step and says nothing here.
    assert_eq!(suggestion.uncaught, 1);
    let below = suggestion.next_below.unwrap();
    assert!(below < 0.3 && legit_low > 0.6);
    let threshold = suggestion.anchor_threshold.unwrap();
    assert!(below < threshold && threshold < legit_low, "between the two");
    assert!(
      (legit_low - threshold - (threshold - below)).abs() < 1e-12,
      "and as far from one as from the other"
    );
    assert!((suggestion.margin.unwrap() - (legit_low - below) / 2.0).abs() < 1e-12);
    assert_eq!(suggestion.recycles_caught, 1);
    assert!(!suggestion.above_threshold, "{threshold} against the overall 0.6");

    // With no recycle that needs it, there is a ceiling and no suggestion.
    let only_edits = replayed(&[line("steady", r#""label": "edit", "#, &steady("indica flower"))]);
    let none = anchor_suggestion(&only_edits, &policy());
    assert!(none.legit_low.is_some() && none.anchor_threshold.is_none());
    assert_eq!((none.uncaught, none.recycles_caught), (0, 0));

    // A truncated history drifted further than it shows, so it places nothing.
    let truncated = replayed(&[line(
      "steady",
      r#""label": "edit", "truncated": true, "#,
      &steady("indica flower"),
    )]);
    assert_eq!(anchor_suggestion(&truncated, &policy()).legit_low, None);
  }

  #[test]
  fn what_the_policy_blocks_and_catches_is_counted_by_rule() {
    let run = run();
    let blocked = blocked(&run, 0);
    assert_eq!((blocked.by_policy.errors, blocked.by_policy.of), (0, 2));
    assert_eq!(blocked.by_policy.rate, Some(0.0));
    assert!(blocked.by_policy.interval.is_some(), "two brands can be resampled");
    assert_eq!((blocked.by_pair_rules.errors, blocked.by_drift_alone.errors), (0, 0));

    let caught = recycles(&run, &policy());
    assert_eq!((caught.at_recycle, caught.later, caught.never), (1, 1, 0));
    assert_eq!(caught.by_drift_alone, 1, "the walk");

    // A drift threshold above how far a legitimate product moves blocks it, and only the drift check does.
    let strict = Policy::new(0.6, Some(0.99)).unwrap();
    let rejudged = {
      let mut run = run.clone();
      for replayed in &mut run.histories {
        for step in &mut replayed.steps {
          step
            .evaluation
            .rejudge(&crate::offline::drift::testing::schema(), &strict);
        }
      }
      run
    };
    let blocked = super::blocked(&rejudged, 0);
    assert_eq!(
      (
        blocked.by_policy.errors,
        blocked.by_pair_rules.errors,
        blocked.by_drift_alone.errors
      ),
      (2, 0, 2)
    );
    assert_eq!(
      recycles(&rejudged, &strict).early,
      1,
      "the walk is now stopped at its legitimate step"
    );
  }

  #[test]
  fn a_quantity_walked_in_small_steps_is_counted_as_one_a_step_limit_cannot_catch() {
    let name = "indica flower";
    let doubling = |product: &str| {
      line(
        product,
        r#""label": "recycle", "#,
        &[
          (name, None, Some(3.5)),
          (name, None, Some(7.0)),
          (name, None, Some(14.0)),
          (name, None, Some(28.0)),
        ],
      )
    };
    let run = replayed(&[
      // A legitimate product whose weight was doubled once, and stayed.
      line(
        "edit",
        r#""label": "edit", "#,
        &[
          (name, None, Some(3.5)),
          (name, None, Some(7.0)),
          (name, None, Some(7.0)),
        ],
      ),
      doubling("walk"),
      // A recycle in one step is a step limit's to catch, and is not a walk.
      line(
        "jump",
        r#""label": "recycle", "#,
        &[(name, None, Some(3.5)), (name, None, Some(28.0))],
      ),
    ]);
    let [weight] = quantity_drift(&run).try_into().unwrap();
    assert_eq!(weight.quantity, "weight");
    assert_eq!(weight.edits, (Some(2.0), Some(2.0)));
    assert_eq!(weight.recycles, (Some(8.0), Some(8.0)));
    assert_eq!(
      weight.walked, 1,
      "eight times the first value in steps of two; the jump is not one"
    );
    assert_eq!(
      (weight.first_incomparable, weight.first_zero_crossings),
      ((0, 0), (0, 0))
    );
  }

  #[test]
  fn the_first_rejection_of_a_history_comes_with_its_reasons() {
    let run = run();
    let (step, reasons) = first_rejection(&run.histories[2]).unwrap();
    assert!(step.step > 2);
    assert!(matches!(reasons, [Reason::AnchorDrift { .. }]), "{reasons:?}");
    assert!(first_rejection(&run.histories[0]).is_none());
  }

  /// A history made by hand: each step's similarity to the anchor, and whether the pair rules pass it. The
  /// configured policy rejects a step the pair rules reject, and nothing else, unless `drift_below` is given.
  fn made(
    id: &str,
    brand_id: Option<&str>,
    label: &str,
    recycled_at: Option<usize>,
    steps: &[(f32, bool)],
    drift_below: Option<f32>,
  ) -> Replayed {
    use similarity_core::score::{Comparison, Evaluation};

    use crate::offline::drift::Step;
    use crate::offline::histories::{History, Version};

    let version = || Version {
      at: None,
      fields: crate::offline::RecordFields::default(),
    };
    Replayed {
      history: History {
        id: id.to_owned(),
        brand_id: brand_id.map(str::to_owned),
        label: match label {
          "edit" => Some(Label::Edit),
          "recycle" => Some(Label::Recycle),
          _ => None,
        },
        label_text: label.to_owned(),
        label_source: None,
        label_confidence: None,
        recycled_at,
        truncated: false,
        original: None,
        versions: (0..=steps.len()).map(|_| version()).collect(),
      },
      steps: steps
        .iter()
        .enumerate()
        .map(|(index, (to_anchor, by_pair_rules))| {
          let mut reasons = Vec::new();
          if !by_pair_rules {
            reasons.push(Reason::BelowThreshold {
              similarity: 0.1,
              threshold: 0.5,
            });
          }
          if let Some(threshold) = drift_below.filter(|threshold| to_anchor < threshold) {
            reasons.push(Reason::AnchorDrift {
              similarity: *to_anchor,
              threshold,
            });
          }
          Step {
            step: index + 1,
            at: None,
            evaluation: Evaluation {
              comparison: Comparison {
                similarity: 0.9,
                fields: Vec::new(),
              },
              anchor_similarity: Some(*to_anchor),
              reasons,
              quantities: Vec::new(),
              measure_changed: false,
            },
            accepted_by_pair_rules: *by_pair_rules,
            anchor_fields: Vec::new(),
            anchor_quantities: Vec::new(),
            written: true,
          }
        })
        .collect(),
    }
  }

  fn run_of(histories: Vec<Replayed>) -> DriftRun {
    DriftRun {
      histories,
      input: crate::offline::pairs::Input::default(),
    }
  }

  /// Where a history was recycled may not be known. All of it is then its recycled part, and none of it can be
  /// called legitimate.
  #[test]
  fn a_recycle_that_does_not_say_where_is_recycled_throughout() {
    let run = run_of(vec![
      made("edit", None, "edit", None, &[(0.75, true)], None),
      made("somewhere", None, "recycle", None, &[(0.5, true), (0.25, true)], None),
    ]);
    let recycled = drift_range(&run, Label::Recycle).unwrap();
    assert_eq!((recycled.count, recycled.lowest), (1, 0.25));
    assert_eq!(
      drift_range(&run, Label::Edit).unwrap().count,
      1,
      "none of the recycle is legitimate"
    );
    let suggestion = anchor_suggestion(&run, &policy());
    assert_eq!((suggestion.uncaught, suggestion.next_below), (1, Some(0.25)));
  }

  #[test]
  fn the_suggestion_is_exactly_halfway_and_counts_only_what_it_separates() {
    // Exact in binary, so that "halfway" and "equal" can be asserted and not approximated.
    let policy = Policy::new(0.5, None).unwrap();
    let run = run_of(vec![
      made("edit", None, "edit", None, &[(0.75, true)], None),
      made("below", None, "recycle", None, &[(0.25, true)], None),
      made("lower", None, "recycle", None, &[(0.125, true)], None),
      // As low as the legitimate history and no lower: no threshold tells the two apart.
      made("level", None, "recycle", None, &[(0.75, true)], None),
      // Caught by the pair rules at its one step: it says nothing about drift.
      made("jump", None, "recycle", None, &[(0.0, false)], None),
    ]);
    let suggestion = anchor_suggestion(&run, &policy);
    assert_eq!(suggestion.legit_low, Some(0.75));
    assert_eq!(suggestion.uncaught, 3);
    assert_eq!(
      suggestion.next_below,
      Some(0.25),
      "the highest that is below, not the one that is level"
    );
    assert_eq!(
      (suggestion.anchor_threshold, suggestion.margin),
      (Some(0.5), Some(0.25))
    );
    assert_eq!(
      suggestion.recycles_caught, 2,
      "the two below; not the level one, and not the jump"
    );
    assert!(
      !suggestion.above_threshold,
      "equal to the overall threshold is not above it"
    );

    let stricter = Policy::new(0.25, None).unwrap();
    assert!(anchor_suggestion(&run, &stricter).above_threshold);
  }

  #[test]
  fn a_share_is_errors_over_histories_and_its_interval_needs_every_history_to_name_one_of_several_brands() {
    let blocked_run = |brands: [Option<&str>; 3]| {
      run_of(vec![
        made("a", brands[0], "edit", None, &[(0.75, false)], None),
        made("b", brands[1], "edit", None, &[(0.75, false)], None),
        made("c", brands[2], "edit", None, &[(0.75, true)], None),
      ])
    };
    let owned = blocked(&blocked_run([Some("b1"), Some("b2"), Some("b3")]), 0);
    assert_eq!((owned.by_policy.errors, owned.by_policy.of), (2, 3));
    assert!((owned.by_policy.rate.unwrap() - 2.0 / 3.0).abs() < 1e-12);
    let interval = owned.by_policy.interval.unwrap();
    assert!(
      interval.low >= 0.0 && interval.high <= 1.0 && interval.low < interval.high,
      "{interval:?}"
    );

    // One history without a brand, and there is nothing sound to resample.
    assert_eq!(
      blocked(&blocked_run([Some("b1"), Some("b2"), None]), 0)
        .by_policy
        .interval,
      None
    );
    // One brand for all of them, likewise.
    assert_eq!(
      blocked(&blocked_run([Some("b1"), Some("b1"), Some("b1")]), 0)
        .by_policy
        .interval,
      None
    );
    assert_eq!(
      blocked(&run_of(Vec::new()), 0).by_policy.rate,
      None,
      "no histories, no share"
    );
  }

  #[test]
  fn a_recycle_is_caught_by_drift_alone_only_if_the_pair_rules_never_reject_it() {
    let policy = Policy::new(0.5, Some(0.5)).unwrap();
    let walk = made(
      "walk",
      None,
      "recycle",
      Some(1),
      &[(0.75, true), (0.25, true)],
      Some(0.5),
    );
    let jump = made("jump", None, "recycle", Some(1), &[(0.25, false)], Some(0.5));
    assert_eq!(recycles(&run_of(vec![walk.clone()]), &policy).by_drift_alone, 1);
    assert_eq!(
      recycles(&run_of(vec![jump.clone()]), &policy).by_drift_alone,
      0,
      "the pair rules caught it too"
    );
    let both = recycles(&run_of(vec![walk, jump]), &policy);
    assert_eq!((both.later, both.at_recycle, both.by_drift_alone), (1, 1, 1));

    // "Caught by drift alone" needs a rejection: a recycle every version of which is accepted was caught by
    // nothing, and one whose every version drifts too far was caught by drift, with no accepted version at all.
    let missed = made(
      "missed",
      None,
      "recycle",
      Some(1),
      &[(0.75, true), (0.75, true)],
      Some(0.5),
    );
    let unaccepted = made("unaccepted", None, "recycle", Some(1), &[(0.25, true)], Some(0.5));
    let missed = recycles(&run_of(vec![missed]), &policy);
    assert_eq!((missed.never, missed.by_drift_alone), (1, 0));
    assert_eq!(recycles(&run_of(vec![unaccepted]), &policy).by_drift_alone, 1);
  }

  #[test]
  fn a_walk_is_further_than_any_edit_goes_in_steps_no_larger_than_an_edit_s() {
    let name = "indica flower";
    let weights = |product: &str, label: &str, grams: &[f64]| {
      let versions: Vec<_> = grams.iter().map(|grams| (name, None, Some(*grams))).collect();
      line(product, &format!(r#""label": "{label}", "#), &versions)
    };
    // The edits double a weight once, and reach four times the first value over two steps.
    let edit = weights("edit", "edit", &[1.0, 2.0, 4.0]);
    let walked =
      |recycle: &[f64]| quantity_drift(&replayed(&[edit.clone(), weights("r", "recycle", recycle)]))[0].walked;
    assert_eq!(walked(&[1.0, 2.0, 4.0, 8.0]), 1, "eight times, in steps of two");
    assert_eq!(
      walked(&[1.0, 8.0]),
      0,
      "eight times in one step is a step limit's to catch"
    );
    assert_eq!(walked(&[1.0, 2.0, 4.0]), 0, "as far as an edit goes, and no further");
    assert_eq!(walked(&[1.0, 2.0]), 0);
  }
}
