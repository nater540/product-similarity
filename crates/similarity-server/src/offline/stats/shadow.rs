//! What a file of shadow events shows.
//!
//! A server with a `[shadow]` table writes what its candidate policy made of live requests (`crate::shadow`).
//! Everything `shadow-report` says about such a file is computed here, from the lines themselves, so that its
//! text and its events cannot disagree.
//!
//! **Two sources, and they are not mixed.** The `shadow_summary` lines count every request the candidate saw,
//! whatever was written about it; the `shadow` lines describe single revisions, and by default only the ones the
//! two policies disagree on. Totals and rates come from the summaries. Everything about *which* revisions (by
//! brand, by reason, by score) comes from the lines. An interval needs to know how many revisions each brand
//! had judged, which only a file written with `events = "all"` holds, so there is one only then.
//!
//! **This measures agreement, not quality.** Nothing here ranks a candidate. A disagreement is a revision to
//! label.

use std::collections::BTreeMap;

use super::{Range, Rate, bootstrap, number};

/// The counts a `shadow_summary` line carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Counts {
  /// Requests the candidate was told about.
  pub observed: u64,
  /// New records: nothing to judge them against.
  pub created: u64,
  /// Revisions both policies judged.
  pub judged: u64,
  /// Revisions both accepted or both rejected.
  pub agreed: u64,
  /// Revisions the live policy accepted and the candidate would have rejected.
  pub only_shadow_rejected: u64,
  /// Revisions the live policy rejected and the candidate would have accepted.
  pub only_live_rejected: u64,
  /// Revisions the candidate could not judge.
  pub failed: u64,
}

/// One judged revision, as a `shadow` line describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Judgment {
  /// What joins it to its reasons.
  pub key: String,
  /// The run of the server that wrote it.
  pub run: Option<u64>,
  /// The record's id.
  pub id: String,
  /// The brand the request named.
  pub brand_id: Option<String>,
  /// Whether the live policy accepted it.
  pub live_accepted: bool,
  /// Whether the candidate would have.
  pub shadow_accepted: bool,
  /// The overall similarity under the live weights.
  pub live_similarity: f64,
  /// The overall similarity under the candidate's.
  pub shadow_similarity: f64,
}

impl Judgment {
  /// Which way the two disagree, if they do.
  pub fn direction(&self) -> Option<Direction> {
    match (self.live_accepted, self.shadow_accepted) {
      (true, false) => Some(Direction::OnlyShadowRejected),
      (false, true) => Some(Direction::OnlyLiveRejected),
      _ => None,
    }
  }
}

/// The two ways a candidate can disagree with the live policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Direction {
  /// The live policy accepted and the candidate would have rejected: the candidate is the stricter here.
  OnlyShadowRejected,
  /// The live policy rejected and the candidate would have accepted.
  OnlyLiveRejected,
}

impl Direction {
  /// The direction as the events spell it.
  pub fn as_str(self) -> &'static str {
    match self {
      Self::OnlyShadowRejected => "only_shadow_rejected",
      Self::OnlyLiveRejected => "only_live_rejected",
    }
  }
}

/// One reason one side gave for rejecting a revision, as a `shadow_reason` line describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct GivenReason {
  /// The revision it belongs to.
  pub key: String,
  /// `live` or `shadow`.
  pub side: String,
  /// The reason's code.
  pub code: String,
  /// What it is about, if it is about one field or quantity.
  pub subject: Option<String>,
}

/// Everything one candidate wrote into a file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShadowLog {
  /// The candidate's name.
  pub shadow: String,
  /// The last summary of each run of the server, by the run's start. A summary that names no run is under `None`.
  pub summaries: BTreeMap<Option<u64>, Counts>,
  /// Every `shadow` line, in file order.
  pub judgments: Vec<Judgment>,
  /// Every `shadow_reason` line, in file order.
  pub reasons: Vec<GivenReason>,
}

/// How many disagreements one brand's revisions account for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrandRow {
  /// The brand, or `None` for requests that named none.
  pub brand_id: Option<String>,
  /// Its revisions that have a line: all that were judged, in a file written with `events = "all"`.
  pub lines: usize,
  /// Of those, the ones only the candidate would have rejected.
  pub only_shadow_rejected: usize,
  /// And the ones only the live policy rejected.
  pub only_live_rejected: usize,
}

/// How often one reason is behind a disagreement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasonRow {
  /// Which way the revision was disagreed on.
  pub direction: Direction,
  /// The reason's code: one the rejecting side gave.
  pub code: String,
  /// What it is about, if it is about one field or quantity.
  pub subject: Option<String>,
  /// How many disagreements carry it.
  pub count: usize,
}

/// The scores of the revisions disagreed on one way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scores {
  /// Which way.
  pub direction: Direction,
  /// Their similarity under the live weights.
  pub live: Range,
  /// And under the candidate's.
  pub shadow: Range,
}

/// What a file of shadow events says about one candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowReport {
  /// The candidate's name.
  pub shadow: String,
  /// How many runs of the server wrote a summary.
  pub runs: usize,
  /// The counts over those runs: the last summary of each, added up. `None` if the file has no summary, in
  /// which case the rates below are over the lines alone.
  pub counts: Option<Counts>,
  /// How many `shadow` lines the file has.
  pub lines: usize,
  /// Whether every judged revision has a line (the file was written with `events = "all"`, and is whole).
  pub complete: bool,
  /// Revisions only the candidate would have rejected, over all that were judged.
  pub only_shadow_rejected: Rate,
  /// Revisions only the live policy rejected, over all that were judged.
  pub only_live_rejected: Rate,
  /// How many brands the intervals were resampled from, when there are intervals.
  pub brands_resampled: Option<usize>,
  /// The brands with the most disagreements, most first.
  pub brands: Vec<BrandRow>,
  /// How many brands have a disagreement, listed or not.
  pub brands_disagreed_on: usize,
  /// The reasons the rejecting side gave for the revisions disagreed on, most frequent first.
  pub reasons: Vec<ReasonRow>,
  /// The scores of the revisions disagreed on, each way.
  pub scores: Vec<Scores>,
}

fn count(value: u64) -> usize {
  usize::try_from(value).unwrap_or(usize::MAX)
}

/// The revisions that have a line, grouped by brand, or `None` if they cannot be resampled by brand: some
/// request named none, or they all named the same one.
fn brands(log: &ShadowLog) -> Option<Vec<Vec<&Judgment>>> {
  let mut groups: BTreeMap<&str, Vec<&Judgment>> = BTreeMap::new();
  for judgment in &log.judgments {
    groups.entry(judgment.brand_id.as_deref()?).or_default().push(judgment);
  }
  (groups.len() > 1).then(|| groups.into_values().collect())
}

fn rate(log: &ShadowLog, direction: Direction, counted: Option<(u64, u64)>, complete: bool, seed: u64) -> Rate {
  let among = |judgments: &mut dyn Iterator<Item = &Judgment>| {
    judgments.fold((0_usize, 0_usize), |(hits, chances), judgment| {
      (
        hits.saturating_add(usize::from(judgment.direction() == Some(direction))),
        chances.saturating_add(1),
      )
    })
  };
  // The summaries count every judged revision; the lines only do in a complete file.
  let (errors, of) = match counted {
    Some((errors, of)) => (count(errors), count(of)),
    None => among(&mut log.judgments.iter()),
  };
  let interval = complete.then(|| brands(log)).flatten().and_then(|clusters| {
    let per_brand: Vec<(usize, usize)> = clusters
      .iter()
      .map(|judgments| among(&mut judgments.iter().copied()))
      .collect();
    bootstrap(&per_brand, seed)
  });
  Rate {
    errors,
    of,
    rate: (of > 0).then(|| number(errors) / number(of)),
    interval,
  }
}

/// The last summary of each run, added up, or `None` if the file has no summary.
fn totals(log: &ShadowLog) -> Option<Counts> {
  (!log.summaries.is_empty()).then(|| {
    log.summaries.values().fold(Counts::default(), |total, run| Counts {
      observed: total.observed.saturating_add(run.observed),
      created: total.created.saturating_add(run.created),
      judged: total.judged.saturating_add(run.judged),
      agreed: total.agreed.saturating_add(run.agreed),
      only_shadow_rejected: total.only_shadow_rejected.saturating_add(run.only_shadow_rejected),
      only_live_rejected: total.only_live_rejected.saturating_add(run.only_live_rejected),
      failed: total.failed.saturating_add(run.failed),
    })
  })
}

/// The brands with a disagreement, most first, cut to `top`, and how many there are in all.
fn brand_rows(log: &ShadowLog, top: usize) -> (Vec<BrandRow>, usize) {
  let mut by_brand: BTreeMap<Option<&str>, BrandRow> = BTreeMap::new();
  for judgment in &log.judgments {
    let row = by_brand
      .entry(judgment.brand_id.as_deref())
      .or_insert_with(|| BrandRow {
        brand_id: judgment.brand_id.clone(),
        lines: 0,
        only_shadow_rejected: 0,
        only_live_rejected: 0,
      });
    row.lines = row.lines.saturating_add(1);
    match judgment.direction() {
      Some(Direction::OnlyShadowRejected) => row.only_shadow_rejected = row.only_shadow_rejected.saturating_add(1),
      Some(Direction::OnlyLiveRejected) => row.only_live_rejected = row.only_live_rejected.saturating_add(1),
      None => {}
    }
  }
  let disagreements = |row: &BrandRow| row.only_shadow_rejected.saturating_add(row.only_live_rejected);
  let mut brand_rows: Vec<BrandRow> = by_brand.into_values().filter(|row| disagreements(row) > 0).collect();
  // Most disagreements first; ties in name order, which is the order they are in.
  brand_rows.sort_by_key(|row| std::cmp::Reverse(disagreements(row)));
  let brands_disagreed_on = brand_rows.len();
  brand_rows.truncate(top);
  (brand_rows, brands_disagreed_on)
}

/// The reasons the rejecting side gave for each revision disagreed on, most frequent first within a direction.
fn reason_rows(log: &ShadowLog) -> Vec<ReasonRow> {
  // The reasons that explain a disagreement are the ones the side that rejected gave.
  let directions: BTreeMap<&str, Direction> = log
    .judgments
    .iter()
    .filter_map(|judgment| Some((judgment.key.as_str(), judgment.direction()?)))
    .collect();
  let mut reason_counts: BTreeMap<(Direction, &str, Option<&str>), usize> = BTreeMap::new();
  for reason in &log.reasons {
    let Some(direction) = directions.get(reason.key.as_str()).copied() else {
      continue;
    };
    let rejecting = match direction {
      Direction::OnlyShadowRejected => "shadow",
      Direction::OnlyLiveRejected => "live",
    };
    if reason.side == rejecting {
      let slot = reason_counts
        .entry((direction, reason.code.as_str(), reason.subject.as_deref()))
        .or_default();
      *slot = slot.saturating_add(1);
    }
  }
  let mut reasons: Vec<ReasonRow> = reason_counts
    .into_iter()
    .map(|((direction, code, subject), count)| ReasonRow {
      direction,
      code: code.to_owned(),
      subject: subject.map(str::to_owned),
      count,
    })
    .collect();
  reasons.sort_by(|a, b| a.direction.cmp(&b.direction).then_with(|| b.count.cmp(&a.count)));
  reasons
}

fn score_ranges(log: &ShadowLog) -> Vec<Scores> {
  [Direction::OnlyShadowRejected, Direction::OnlyLiveRejected]
    .into_iter()
    .filter_map(|direction| {
      let of = |pick: fn(&Judgment) -> f64| {
        Range::of(
          log
            .judgments
            .iter()
            .filter(|judgment| judgment.direction() == Some(direction))
            .map(pick),
        )
      };
      Some(Scores {
        direction,
        live: of(|judgment| judgment.live_similarity)?,
        shadow: of(|judgment| judgment.shadow_similarity)?,
      })
    })
    .collect()
}

/// Reports on one candidate's events. `top` is how many brands to list; `seed` fixes the resampling.
pub fn report(log: &ShadowLog, top: usize, seed: u64) -> ShadowReport {
  let counts = totals(log);
  // Without a summary there is nothing to hold the lines against, and they are taken for all there is.
  let complete = counts.is_none_or(|counts| count(counts.judged) == log.judgments.len());
  let (brands_listed, brands_disagreed_on) = brand_rows(log, top);
  let counted = |pick: fn(&Counts) -> u64| counts.as_ref().map(|counts| (pick(counts), counts.judged));
  ShadowReport {
    shadow: log.shadow.clone(),
    runs: log.summaries.len(),
    counts,
    lines: log.judgments.len(),
    complete,
    only_shadow_rejected: rate(
      log,
      Direction::OnlyShadowRejected,
      counted(|counts| counts.only_shadow_rejected),
      complete,
      seed,
    ),
    only_live_rejected: rate(
      log,
      Direction::OnlyLiveRejected,
      counted(|counts| counts.only_live_rejected),
      complete,
      seed,
    ),
    brands_resampled: complete.then(|| brands(log)).flatten().map(|clusters| clusters.len()),
    brands: brands_listed,
    brands_disagreed_on,
    reasons: reason_rows(log),
    scores: score_ranges(log),
  }
}
