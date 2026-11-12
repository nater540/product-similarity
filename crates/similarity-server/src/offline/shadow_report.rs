//! Reading the events a shadowed server wrote, and saying what is in them.
//!
//! The input is the server's stdout: the `shadow*` result lines of `crate::shadow`, among whatever else was
//! written there. Lines of any other kind are passed over. Every number in the report and in the events comes
//! from [`stats::shadow`](super::stats::shadow).
//!
//! The report cannot say whether a candidate is better. It says how often the two policies disagree, in which
//! direction, for whom, and why, and it can write the revisions they disagree on as a list to be labeled
//! ([`ids`]). Labeled, those go through `calibrate` like any other pairs, and that is where a candidate is judged.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use similarity_log::RESULT;

use super::events::six_wide;
use super::stats::shadow::{Counts, Direction, GivenReason, Judgment, ShadowLog, ShadowReport};
use super::stats::{Range, Rate};

#[derive(Deserialize)]
struct JudgmentLine {
  key: String,
  #[serde(default)]
  run: Option<u64>,
  id: String,
  #[serde(default)]
  tenant: Option<String>,
  shadow: String,
  live_verdict: String,
  shadow_verdict: String,
  live_similarity: f64,
  shadow_similarity: f64,
}

#[derive(Deserialize)]
struct ReasonLine {
  key: String,
  shadow: String,
  side: String,
  code: String,
  #[serde(default)]
  subject: Option<String>,
}

#[derive(Deserialize)]
struct SummaryLine {
  shadow: String,
  #[serde(default)]
  run: Option<u64>,
  observed: u64,
  created: u64,
  judged: u64,
  agreed: u64,
  only_shadow_rejected: u64,
  only_live_rejected: u64,
  failed: u64,
}

fn accepted(verdict: &str) -> anyhow::Result<bool> {
  match verdict {
    "accepted" => Ok(true),
    "rejected" => Ok(false),
    other => bail!("`{other}` is not a verdict a shadow writes"),
  }
}

fn log_of(logs: &mut BTreeMap<String, ShadowLog>, name: String) -> &mut ShadowLog {
  logs.entry(name.clone()).or_insert_with(|| ShadowLog {
    shadow: name,
    ..ShadowLog::default()
  })
}

/// Reads shadow events, by candidate. Result lines of other kinds, and lines that are not results, are passed
/// over; a later summary of a run replaces an earlier one, since each counts from the server's start.
///
/// # Errors
///
/// Returns an error naming the line if one is not a JSON object, or is a shadow line this cannot read.
pub fn read(ndjson: &str) -> anyhow::Result<BTreeMap<String, ShadowLog>> {
  let mut logs: BTreeMap<String, ShadowLog> = BTreeMap::new();
  for (index, line) in ndjson.lines().enumerate().filter(|(_, line)| !line.trim().is_empty()) {
    let number = index.saturating_add(1);
    let value: Value = serde_json::from_str(line).with_context(|| format!("line {number} is not JSON"))?;
    let Some(object) = value.as_object() else {
      bail!("line {number} is not a JSON object");
    };
    if object.get("target").and_then(Value::as_str) != Some(RESULT) {
      continue;
    }
    let invalid = || format!("line {number} is not a shadow line this version reads");
    match object.get("message").and_then(Value::as_str) {
      Some("shadow") => {
        let parsed: JudgmentLine = serde_json::from_value(value).with_context(invalid)?;
        let judgment = Judgment {
          live_accepted: accepted(&parsed.live_verdict).with_context(invalid)?,
          shadow_accepted: accepted(&parsed.shadow_verdict).with_context(invalid)?,
          key: parsed.key,
          run: parsed.run,
          id: parsed.id,
          tenant: parsed.tenant,
          live_similarity: parsed.live_similarity,
          shadow_similarity: parsed.shadow_similarity,
        };
        log_of(&mut logs, parsed.shadow).judgments.push(judgment);
      }
      Some("shadow_reason") => {
        let parsed: ReasonLine = serde_json::from_value(value).with_context(invalid)?;
        log_of(&mut logs, parsed.shadow).reasons.push(GivenReason {
          key: parsed.key,
          side: parsed.side,
          code: parsed.code,
          subject: parsed.subject,
        });
      }
      Some("shadow_summary") => {
        let parsed: SummaryLine = serde_json::from_value(value).with_context(invalid)?;
        let counts = Counts {
          observed: parsed.observed,
          created: parsed.created,
          judged: parsed.judged,
          agreed: parsed.agreed,
          only_shadow_rejected: parsed.only_shadow_rejected,
          only_live_rejected: parsed.only_live_rejected,
          failed: parsed.failed,
        };
        log_of(&mut logs, parsed.shadow).summaries.insert(parsed.run, counts);
      }
      _ => {}
    }
  }
  Ok(logs)
}

/// The one candidate a report is about: the one named, or the only one in the file.
///
/// # Errors
///
/// Returns an error if the file has no shadow lines, has none for `name`, or has several candidates and none was
/// named.
pub fn select(mut logs: BTreeMap<String, ShadowLog>, name: Option<&str>) -> anyhow::Result<ShadowLog> {
  let names = || {
    logs
      .keys()
      .map(|name| format!("`{name}`"))
      .collect::<Vec<_>>()
      .join(", ")
  };
  match name {
    Some(name) => {
      let listed = names();
      logs
        .remove(name)
        .with_context(|| format!("the file has no lines of a shadow called `{name}`; it has {listed}"))
    }
    None if logs.len() > 1 => bail!(
      "the file has lines of several shadows ({}); say which with `--shadow`",
      names()
    ),
    None => logs
      .into_values()
      .next()
      .context("the file has no shadow lines: is it the stdout of a server with a `[shadow]` table?"),
  }
}

fn percent(rate: &Rate) -> String {
  let Some(value) = rate.rate else {
    return "-".to_owned();
  };
  let interval = rate.interval.map_or_else(String::new, |interval| {
    format!(" ({:.2}% to {:.2}%)", interval.low * 100.0, interval.high * 100.0)
  });
  format!("{:.2}%{interval}", value * 100.0)
}

fn range(range: Range) -> String {
  format!("{:.3} to {:.3}", range.low, range.high)
}

fn direction_text(direction: Direction) -> &'static str {
  match direction {
    Direction::OnlyShadowRejected => "live accepted, the candidate would have rejected",
    Direction::OnlyLiveRejected => "live rejected, the candidate would have accepted",
  }
}

fn render_tenants(report: &ShadowReport, out: &mut String) {
  if !report.tenants.is_empty() {
    let _ = writeln!(
      out,
      "\n{:<28} {:>8} {:>22} {:>20}",
      "tenant", "lines", "only candidate rejects", "only live rejects"
    );
    for row in &report.tenants {
      let _ = writeln!(
        out,
        "{:<28} {:>8} {:>22} {:>20}",
        row.tenant.as_deref().unwrap_or("(none named)"),
        row.lines,
        row.only_shadow_rejected,
        row.only_live_rejected
      );
    }
    let more = report.tenants_disagreed_on.saturating_sub(report.tenants.len());
    if more > 0 {
      let _ = writeln!(out, "and {more} more tenant(s) with a disagreement");
    }
  }
}

/// Renders a report for a terminal. Every number is from `report`.
pub fn render(report: &ShadowReport) -> String {
  let mut out = String::new();
  // Writing to a `String` cannot fail, here and below.
  let _ = writeln!(out, "shadow    {}", report.shadow);
  match &report.counts {
    Some(counts) => {
      let _ = writeln!(
        out,
        "seen      {} requests over {} run(s) of the server: {} new records, {} revisions judged, {} the candidate \
         could not judge",
        counts.observed, report.runs, counts.created, counts.judged, counts.failed
      );
      let _ = writeln!(out, "agreed    {} of {}", counts.agreed, counts.judged);
    }
    None => {
      let _ = writeln!(
        out,
        "seen      no summary line is in the file, so the lines are all there is to count: {}",
        report.lines
      );
    }
  }
  for (direction, rate) in [
    (Direction::OnlyShadowRejected, &report.only_shadow_rejected),
    (Direction::OnlyLiveRejected, &report.only_live_rejected),
  ] {
    let _ = writeln!(
      out,
      "disagreed {} of {}  {}  {}",
      rate.errors,
      rate.of,
      percent(rate),
      direction_text(direction)
    );
  }
  match report.tenants_resampled {
    Some(tenants) => {
      let _ = writeln!(
        out,
        "          intervals are the middle 95% over resamples of {tenants} tenants"
      );
    }
    None if !report.complete => {
      let _ = writeln!(
        out,
        "          no intervals: the file has a line for {} of the judged revisions (`events = \"all\"` writes one \
         for each)",
        report.lines
      );
    }
    None => {
      let _ = writeln!(
        out,
        "          no intervals: they need every revision to name one of several tenants"
      );
    }
  }

  for scores in &report.scores {
    let _ = writeln!(
      out,
      "\n{}\n  scored {} by the live weights and {} by the candidate's",
      direction_text(scores.direction),
      range(scores.live),
      range(scores.shadow)
    );
    for reason in report
      .reasons
      .iter()
      .filter(|reason| reason.direction == scores.direction)
    {
      let about = reason
        .subject
        .as_ref()
        .map_or_else(String::new, |subject| format!(" {subject}"));
      let _ = writeln!(out, "  {:>6}  {}{about}", reason.count, reason.code);
    }
  }

  render_tenants(report, &mut out);
  let _ = writeln!(
    out,
    "\nA disagreement says the two policies differ on a revision, not which is right. Label the revisions \
     (`--ids`), and judge the candidate with `calibrate`."
  );
  out
}

/// One disagreement, as it is handed over for labeling.
#[derive(Debug, Serialize)]
struct Disagreement<'a> {
  key: &'a str,
  id: &'a str,
  #[serde(skip_serializing_if = "Option::is_none")]
  tenant: Option<&'a str>,
  live_verdict: &'static str,
  shadow_verdict: &'static str,
  live_similarity: f64,
  shadow_similarity: f64,
}

/// The revisions the two policies disagree on, one JSON object per line, in file order: what to look up and
/// label. An id is there once for each revision of it that was disagreed on.
///
/// # Errors
///
/// Returns an error only if a line cannot be written as JSON, which these cannot fail to be.
pub fn ids(log: &ShadowLog) -> anyhow::Result<String> {
  let verdict = |accepted: bool| if accepted { "accepted" } else { "rejected" };
  let mut out = String::new();
  for judgment in log.judgments.iter().filter(|judgment| judgment.direction().is_some()) {
    let line = serde_json::to_string(&Disagreement {
      key: &judgment.key,
      id: &judgment.id,
      tenant: judgment.tenant.as_deref(),
      live_verdict: verdict(judgment.live_accepted),
      shadow_verdict: verdict(judgment.shadow_accepted),
      live_similarity: judgment.live_similarity,
      shadow_similarity: judgment.shadow_similarity,
    })
    .context("a disagreement does not serialize")?;
    out.push_str(&line);
    out.push('\n');
  }
  Ok(out)
}

/// Writes a report as events: one `shadow_report` line, then one line per tenant, reason, and direction listed.
pub fn emit(report: &ShadowReport) {
  let shadow = report.shadow.as_str();
  let counts = report.counts.as_ref();
  let interval = |rate: &Rate| {
    (
      rate.interval.map(|interval| six_wide(interval.low)),
      rate.interval.map(|interval| six_wide(interval.high)),
    )
  };
  let (shadow_low, shadow_high) = interval(&report.only_shadow_rejected);
  let (live_low, live_high) = interval(&report.only_live_rejected);
  tracing::info!(
    target: RESULT,
    shadow,
    runs = report.runs,
    observed = counts.map(|counts| counts.observed),
    created = counts.map(|counts| counts.created),
    judged = report.only_shadow_rejected.of,
    agreed = counts.map(|counts| counts.agreed),
    failed = counts.map(|counts| counts.failed),
    lines = report.lines,
    complete = report.complete,
    only_shadow_rejected = report.only_shadow_rejected.errors,
    only_shadow_rejected_rate = report.only_shadow_rejected.rate.map(six_wide),
    only_shadow_rejected_low = shadow_low,
    only_shadow_rejected_high = shadow_high,
    only_live_rejected = report.only_live_rejected.errors,
    only_live_rejected_rate = report.only_live_rejected.rate.map(six_wide),
    only_live_rejected_low = live_low,
    only_live_rejected_high = live_high,
    tenants_resampled = report.tenants_resampled,
    tenants_disagreed_on = report.tenants_disagreed_on,
    "shadow_report"
  );
  for scores in &report.scores {
    tracing::info!(
      target: RESULT,
      shadow,
      direction = scores.direction.as_str(),
      count = scores.live.count,
      live_similarity_low = six_wide(scores.live.low),
      live_similarity_high = six_wide(scores.live.high),
      shadow_similarity_low = six_wide(scores.shadow.low),
      shadow_similarity_high = six_wide(scores.shadow.high),
      "shadow_report_scores"
    );
  }
  for reason in &report.reasons {
    tracing::info!(
      target: RESULT,
      shadow,
      direction = reason.direction.as_str(),
      code = reason.code.as_str(),
      subject = reason.subject.as_deref(),
      count = reason.count,
      "shadow_report_reason"
    );
  }
  for row in &report.tenants {
    tracing::info!(
      target: RESULT,
      shadow,
      tenant = row.tenant.as_deref(),
      lines = row.lines,
      only_shadow_rejected = row.only_shadow_rejected,
      only_live_rejected = row.only_live_rejected,
      "shadow_report_tenant"
    );
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use similarity_core::{FieldSpec, Policy, Schema};
  use similarity_store::MemoryStore;

  use super::super::stats::shadow::report;
  use super::*;
  use crate::config::ShadowEvents;
  use crate::engine::{Engine, EvaluateInput, RecordInput, store_layout};
  use crate::shadow::{PolicyShadow, Summary};
  use crate::worker::TextEmbedder;
  use crate::worker::testing::WordBagEmbedder;

  const NAME: &str = "blueberry gummies vegan fruit chews";
  /// One word of five is different: 0.8 on the name.
  const RENAMED: &str = "blueberry gummies vegan fruit cartridge";
  /// Nothing in common.
  const REPLACED: &str = "sour diesel resin vape pen";

  fn request(id: &str, tenant: Option<&str>, name: &str) -> EvaluateInput {
    EvaluateInput {
      record: RecordInput {
        id: Some(id.to_owned()),
        tenant: tenant.map(str::to_owned),
        fields: vec![("name".to_owned(), name.to_owned())],
        quantities: Vec::new(),
      },
      ..EvaluateInput::default()
    }
  }

  /// What a server with a shadow writes for `requests`: its stdout, and the shadow's own counts at the end. The
  /// live policy accepts at 0.5 and the candidate at 0.9, so a one-word rename is a disagreement.
  fn written(events: ShadowEvents, requests: &[EvaluateInput]) -> (String, Summary) {
    let schema = Schema::new(vec![FieldSpec {
      name: "name".to_owned(),
      weight: 1.0,
      required: true,
      reject_below: None,
    }])
    .unwrap();
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema, &embedder);
    let live = Policy::new(0.5, None).unwrap();
    let shadow = Arc::new(PolicyShadow::new(
      "candidate",
      schema.clone(),
      Policy::new(0.9, None).unwrap(),
      embedder.dimension(),
      events,
    ));
    let engine =
      Engine::new(MemoryStore::new(layout, dimension), embedder, schema.clone(), live).with_observer(shadow.clone());
    let captured = similarity_log::testing::capture("info", || {
      shadow.announce(&schema, &live);
      let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
      runtime.block_on(async {
        for request in requests {
          engine.evaluate(request.clone()).await.unwrap();
        }
      });
      // On a timer, and once more as the server stops: the later line is the one that counts.
      shadow.emit_summary();
      shadow.emit_summary();
    });
    (captured.results.join("\n"), shadow.summary())
  }

  /// Three tenants. `brand-1` renames two records (two disagreements) and resaves one; `brand-2` renames one
  /// and recycles one (rejected by both); `brand-3` only resaves.
  fn traffic() -> Vec<EvaluateInput> {
    let mut requests = Vec::new();
    for (id, tenant) in [
      ("a", "brand-1"),
      ("b", "brand-1"),
      ("c", "brand-2"),
      ("d", "brand-2"),
      ("e", "brand-3"),
    ] {
      requests.push(request(id, Some(tenant), NAME));
    }
    requests.push(request("a", Some("brand-1"), RENAMED));
    requests.push(request("b", Some("brand-1"), RENAMED));
    requests.push(request("b", Some("brand-1"), RENAMED));
    requests.push(request("c", Some("brand-2"), RENAMED));
    requests.push(request("d", Some("brand-2"), REPLACED));
    requests.push(request("e", Some("brand-3"), NAME));
    requests
  }

  fn only(ndjson: &str) -> ShadowLog {
    select(read(ndjson).unwrap(), None).unwrap()
  }

  /// The reader and the writer are held to each other: what a shadow writes, the report reads, and its counts
  /// are the shadow's own.
  #[test]
  fn a_report_reads_what_a_shadow_wrote_and_counts_what_the_shadow_counted() {
    let (ndjson, summary) = written(ShadowEvents::All, &traffic());
    let log = only(&ndjson);
    assert_eq!(log.shadow, "candidate");
    assert_eq!(log.summaries.len(), 1, "two summaries of one run: the later one stands");
    let made = report(&log, 10, 0);
    let counts = made.counts.unwrap();
    assert_eq!(
      (
        counts.observed,
        counts.created,
        counts.judged,
        counts.agreed,
        counts.failed
      ),
      (
        summary.observed,
        summary.created,
        summary.judged,
        summary.agreed,
        summary.failed
      )
    );
    assert_eq!((counts.observed, counts.created, counts.judged), (11, 5, 6));
    assert_eq!((made.runs, made.lines, made.complete), (1, 6, true));

    assert_eq!((made.only_shadow_rejected.errors, made.only_shadow_rejected.of), (3, 6));
    assert_eq!((made.only_live_rejected.errors, made.only_live_rejected.of), (0, 6));
    assert_eq!(made.only_shadow_rejected.rate, Some(0.5));
    let interval = made.only_shadow_rejected.interval.unwrap();
    assert!(interval.low <= 0.5 && interval.high >= 0.5, "{interval:?}");
    assert_eq!(made.tenants_resampled, Some(3));

    // Most disagreements first; a tenant with none is not listed.
    let rows: Vec<(Option<&str>, usize, usize, usize)> = made
      .tenants
      .iter()
      .map(|row| {
        (
          row.tenant.as_deref(),
          row.lines,
          row.only_shadow_rejected,
          row.only_live_rejected,
        )
      })
      .collect();
    assert_eq!(rows, [(Some("brand-1"), 3, 2, 0), (Some("brand-2"), 2, 1, 0)]);
    assert_eq!(made.tenants_disagreed_on, 2);
    assert_eq!(report(&log, 1, 0).tenants.len(), 1);
    assert_eq!(
      report(&log, 1, 0).tenants_disagreed_on,
      2,
      "cut to one, and still two of them"
    );

    // Why: the candidate's reason for each of the three, and none of the reasons both gave for the recycle.
    assert_eq!(made.reasons.len(), 1);
    let reason = &made.reasons[0];
    assert_eq!(
      (
        reason.direction,
        reason.code.as_str(),
        reason.subject.as_deref(),
        reason.count
      ),
      (Direction::OnlyShadowRejected, "below_threshold", None, 3)
    );
    assert_eq!(made.scores.len(), 1);
    let scores = made.scores[0];
    assert_eq!(
      (scores.direction, scores.live.count),
      (Direction::OnlyShadowRejected, 3)
    );
    assert!(
      (scores.live.low - 0.8).abs() < 1e-5 && (scores.shadow.high - 0.8).abs() < 1e-5,
      "{scores:?}"
    );
  }

  /// By default a shadow writes a line only for a disagreement. The totals still come out, from the summary; what
  /// cannot is an interval, which needs to know how many revisions each tenant had judged.
  #[test]
  fn with_only_disagreements_written_the_rates_are_the_summarys_and_have_no_interval() {
    let (ndjson, summary) = written(ShadowEvents::Disagreements, &traffic());
    let made = report(&only(&ndjson), 10, 0);
    assert_eq!((made.lines, made.complete), (3, false));
    assert_eq!(
      (made.only_shadow_rejected.errors, made.only_shadow_rejected.of),
      (3, usize::try_from(summary.judged).unwrap())
    );
    assert_eq!(made.only_shadow_rejected.rate, Some(0.5));
    assert_eq!(
      (made.only_shadow_rejected.interval, made.tenants_resampled),
      (None, None)
    );
    assert_eq!(made.tenants[0].lines, 2, "only the lines that were written");
    let text = render(&made);
    assert!(
      text.contains("no intervals: the file has a line for 3 of the judged revisions"),
      "{text}"
    );
  }

  fn summary_line(shadow: &str, run: u64, judged: u64, only_shadow: u64) -> String {
    format!(
      r#"{{"message":"shadow_summary","shadow":"{shadow}","kind":"policy","run":{run},"observed":{},"created":1,"judged":{judged},"agreed":{},"only_shadow_rejected":{only_shadow},"only_live_rejected":0,"failed":0,"target":"result"}}"#,
      judged + 1,
      judged - only_shadow
    )
  }

  /// Each summary counts from its server's start, so a file that several runs wrote into is the last summary of
  /// each, added up.
  #[test]
  fn several_runs_of_a_server_are_added_up_from_the_last_summary_of_each() {
    let lines = [
      summary_line("candidate", 100, 10, 1),
      summary_line("candidate", 100, 40, 4),
      summary_line("candidate", 200, 5, 0),
      summary_line("candidate", 200, 60, 6),
    ]
    .join("\n");
    let made = report(&only(&lines), 10, 0);
    let counts = made.counts.unwrap();
    assert_eq!(
      (made.runs, counts.judged, counts.only_shadow_rejected, counts.agreed),
      (2, 100, 10, 90)
    );
    assert_eq!((counts.observed, counts.created), (102, 2));
    assert_eq!(
      (
        made.only_shadow_rejected.errors,
        made.only_shadow_rejected.of,
        made.only_shadow_rejected.rate
      ),
      (10, 100, Some(0.1))
    );
    assert!(!made.complete, "a hundred judged and no line for any");
  }

  #[test]
  fn a_file_with_no_summary_is_reported_from_its_lines_alone_and_says_so() {
    let (ndjson, _) = written(ShadowEvents::All, &traffic());
    let lines: Vec<&str> = ndjson.lines().filter(|line| !line.contains("shadow_summary")).collect();
    let made = report(&only(&lines.join("\n")), 10, 0);
    assert_eq!((made.counts, made.runs, made.lines), (None, 0, 6));
    assert_eq!((made.only_shadow_rejected.errors, made.only_shadow_rejected.of), (3, 6));
    assert!(render(&made).contains("no summary line is in the file"));
  }

  #[test]
  fn other_lines_are_passed_over_and_a_file_of_several_shadows_needs_one_named() {
    let (ndjson, _) = written(ShadowEvents::All, &traffic());
    let noisy = format!(
      "{}\n\n{}\n{ndjson}\n{}",
      r#"{"level":"INFO","message":"evaluate","id":"a","target":"record_similarity"}"#,
      r#"{"message":"load_phase","phase":"edit","target":"result"}"#,
      summary_line("other", 300, 4, 2)
    );
    let logs = read(&noisy).unwrap();
    assert_eq!(
      logs.keys().map(String::as_str).collect::<Vec<_>>(),
      ["candidate", "other"]
    );
    assert_eq!(
      logs["candidate"],
      only(&ndjson),
      "what is not a shadow line changes nothing"
    );

    let unnamed = select(logs.clone(), None).unwrap_err().to_string();
    assert!(
      unnamed.contains("`candidate`, `other`") && unnamed.contains("--shadow"),
      "{unnamed}"
    );
    assert_eq!(select(logs.clone(), Some("other")).unwrap().summaries.len(), 1);
    let missing = select(logs, Some("nobody")).unwrap_err().to_string();
    assert!(missing.contains("no lines of a shadow called `nobody`"), "{missing}");

    let empty = select(
      read(r#"{"message":"evaluate","target":"record_similarity"}"#).unwrap(),
      None,
    )
    .unwrap_err();
    assert!(empty.to_string().contains("no shadow lines"), "{empty}");
    let broken = format!("{ndjson}\nnot json");
    assert!(format!("{:#}", read(&broken).unwrap_err()).contains("line 16 is not JSON"));
    let odd = r#"{"message":"shadow","key":"1-1","id":"a","shadow":"s","live_verdict":"created","shadow_verdict":"accepted","live_similarity":1.0,"shadow_similarity":1.0,"target":"result"}"#;
    assert!(format!("{:#}", read(odd).unwrap_err()).contains("line 1 is not a shadow line"));
  }

  /// The list for labeling: every revision disagreed on, in file order, and nothing that was agreed on.
  #[test]
  fn the_revisions_disagreed_on_are_listed_for_labeling() {
    let (ndjson, _) = written(ShadowEvents::All, &traffic());
    let listed = ids(&only(&ndjson)).unwrap();
    let lines: Vec<Value> = listed.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    let seen: Vec<(&str, &str, &str, &str)> = lines
      .iter()
      .map(|line| {
        (
          line["id"].as_str().unwrap(),
          line["tenant"].as_str().unwrap(),
          line["live_verdict"].as_str().unwrap(),
          line["shadow_verdict"].as_str().unwrap(),
        )
      })
      .collect();
    assert_eq!(
      seen,
      [
        ("a", "brand-1", "accepted", "rejected"),
        ("b", "brand-1", "accepted", "rejected"),
        ("c", "brand-2", "accepted", "rejected"),
      ]
    );
    assert!(lines.iter().all(|line| line["key"].as_str().unwrap().contains('-')));
    assert!((lines[0]["live_similarity"].as_f64().unwrap() - 0.8).abs() < 1e-5);
    assert!(listed.ends_with('\n') && !listed.contains("gummies"));
  }

  /// The text and the events are two renderings of one report.
  #[test]
  fn the_shadow_report_and_its_events_give_the_same_numbers() {
    let (ndjson, _) = written(ShadowEvents::All, &traffic());
    let made = report(&only(&ndjson), 10, 0);
    let text = render(&made);
    assert!(text.starts_with("shadow    candidate\nseen      11 requests over 1 run(s) of the server: 5 new records, 6 revisions judged, 0 the candidate could not judge\nagreed    3 of 6\n"), "{text}");
    assert!(text.contains("disagreed 3 of 6  50.00% ("), "{text}");
    assert!(text.contains("disagreed 0 of 6  0.00% ("), "{text}");
    assert!(
      text.contains("intervals are the middle 95% over resamples of 3 tenants"),
      "{text}"
    );
    assert!(text.contains("       3  below_threshold"), "{text}");
    assert!(
      text.contains("brand-1                             3                      2                    0"),
      "{text}"
    );
    assert!(text.contains("not which is right"), "{text}");

    let captured = similarity_log::testing::capture("off", || emit(&made));
    let events: Vec<Value> = captured
      .results
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();
    for event in &events {
      for (key, value) in event.as_object().unwrap() {
        assert!(
          !value.is_object() && !value.is_array() && !value.is_null(),
          "`{key}` in {event}"
        );
      }
    }
    let messages: Vec<&str> = events.iter().map(|event| event["message"].as_str().unwrap()).collect();
    assert_eq!(
      messages,
      [
        "shadow_report",
        "shadow_report_scores",
        "shadow_report_reason",
        "shadow_report_tenant",
        "shadow_report_tenant"
      ]
    );
    let line = &events[0];
    assert_eq!(
      (
        line["judged"].as_u64(),
        line["agreed"].as_u64(),
        line["only_shadow_rejected"].as_u64(),
        line["only_live_rejected"].as_u64(),
        line["lines"].as_u64(),
        line["complete"].as_bool(),
        line["tenants_resampled"].as_u64(),
      ),
      (Some(6), Some(3), Some(3), Some(0), Some(6), Some(true), Some(3))
    );
    let interval = made.only_shadow_rejected.interval.unwrap();
    assert!((line["only_shadow_rejected_low"].as_f64().unwrap() - interval.low).abs() < 1e-6);
    assert_eq!(
      (events[3]["tenant"].as_str(), events[3]["only_shadow_rejected"].as_u64()),
      (Some("brand-1"), Some(2))
    );
  }
}
