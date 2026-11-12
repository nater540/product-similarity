//! A candidate policy, judged in the shadow of the live one.
//!
//! `calibrate`, `calibrate-diff`, and `calibrate-sweep` say what another threshold, weight, floor, or limit makes
//! of labeled pairs. They cannot say what it would make of the requests the service actually gets. A
//! [`PolicyShadow`] does: it is told about every judged request (it is an [`Observer`] of the engine), judges the
//! same vectors against the same baseline under the candidate's schema and policy, and writes down where the
//! candidate would have decided the other way.
//!
//! **A shadow decides nothing.** It is told about a request after the live outcome is final and returns nothing,
//! so it cannot change a verdict, a response, or a write, whatever it concludes and even if it fails. What it
//! costs the live path is one more weighted comparison of vectors that are already in memory, and a line written
//! as a log line is where there is one to write: no model, no store, nothing awaited.
//!
//! **A shadow measures agreement, not quality.** A disagreement says the two policies differ on a request, not
//! which of them is right; only a label can say that. The ids it lists are for labeling, and the labeled pairs
//! then go through `calibrate` like any others.
//!
//! The candidate judges against the baseline the live policy built: the history the service really accepted. A
//! revision the candidate would have refused still becomes the baseline if the live policy stored it, so every
//! judgment answers "what would the candidate have said to this request, here", which is the question a change
//! of policy raises on the day it is made.
//!
//! What it writes are results (`similarity_log::RESULT`: stdout, never filtered), flat like the calibration's
//! events (D17), and their names are an interface in the same way:
//!
//! - `shadow_started`, once, with the candidate's policy beside the live one, then one `shadow_schema_field` and
//!   `shadow_schema_quantity` line per item, so that a file of shadow events says what produced it;
//! - `shadow`, one per disagreement (or per judged request, with `events = "all"`), and one `shadow_reason` per
//!   reason either side gave, joined on `key`;
//! - `shadow_summary`, the counts since the server started, on a timer and once more as the server stops.
//!
//! A line names a record by its id and the tenant the request named, and carries scores and codes: never a
//! record's text.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use similarity_core::score::{self, Evaluation};
use similarity_core::{Policy, Reason, Schema};
use similarity_log::RESULT;

use crate::config::{Config, ShadowEvents};
use crate::engine::{Judged, Observation, Observer, WriteMode};
use crate::offline::events::{six, six_wide};

/// What every line of a policy shadow carries as its `kind`.
const KIND: &str = "policy";

#[derive(Debug, Default)]
struct Counters {
  observed: AtomicU64,
  created: AtomicU64,
  judged: AtomicU64,
  agreed: AtomicU64,
  only_shadow_rejected: AtomicU64,
  only_live_rejected: AtomicU64,
  failed: AtomicU64,
}

/// What a shadow has seen since it was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Summary {
  /// Requests it was told about: `created`, `judged`, and `failed` together.
  pub observed: u64,
  /// New records. There was nothing to judge them against, so the candidate has no opinion of them.
  pub created: u64,
  /// Revisions both policies judged: `agreed` and the two kinds of disagreement together.
  pub judged: u64,
  /// Revisions both accepted or both rejected.
  pub agreed: u64,
  /// Revisions the live policy accepted and the candidate would have rejected.
  pub only_shadow_rejected: u64,
  /// Revisions the live policy rejected and the candidate would have accepted.
  pub only_live_rejected: u64,
  /// Revisions the candidate could not judge. Not expected: it reads the vectors the live policy just scored.
  pub failed: u64,
}

/// A candidate schema and policy that judge what the live ones judged.
#[derive(Debug)]
pub struct PolicyShadow {
  name: String,
  schema: Schema,
  policy: Policy,
  dimension: usize,
  events: ShadowEvents,
  /// When this shadow was made, in seconds since the epoch: the first half of every `key`, so that keys from
  /// two runs of the server do not collide in one file.
  started: u64,
  counters: Counters,
}

fn bump(counter: &AtomicU64) -> u64 {
  // The count after this one. A counter that wrapped would take half a million years of requests.
  counter.fetch_add(1, Ordering::Relaxed).wrapping_add(1)
}

fn verdict(accepted: bool) -> &'static str {
  if accepted { "accepted" } else { "rejected" }
}

fn mode_name(mode: WriteMode) -> &'static str {
  match mode {
    WriteMode::Enforce => "enforce",
    WriteMode::DryRun => "dry_run",
    WriteMode::Force => "force",
  }
}

fn events_name(events: ShadowEvents) -> &'static str {
  match events {
    ShadowEvents::Disagreements => "disagreements",
    ShadowEvents::All => "all",
  }
}

impl PolicyShadow {
  /// A shadow called `name` that judges by `schema` and `policy`.
  ///
  /// `schema` must name the fields the live schema names (`Config::shadowed` makes one that does), and
  /// `dimension` is the live embedder's: the candidate reads the live vectors.
  pub fn new(name: impl Into<String>, schema: Schema, policy: Policy, dimension: usize, events: ShadowEvents) -> Self {
    Self {
      name: name.into(),
      schema,
      policy,
      dimension,
      events,
      started: SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs()),
      counters: Counters::default(),
    }
  }

  /// The shadow `config` asks for, or `None` if it asks for none. `dimension` is the live embedder's.
  ///
  /// # Errors
  ///
  /// Returns an error if `[shadow]` does not describe a valid schema and policy, which `Config::parse` has
  /// already ruled out for a configuration it returned.
  pub fn from_config(config: &Config, dimension: usize) -> anyhow::Result<Option<Self>> {
    let Some(shadow) = &config.shadow else {
      return Ok(None);
    };
    let candidate = config.shadowed()?;
    Ok(Some(Self::new(
      shadow.name.clone(),
      candidate.schema()?,
      candidate.policy()?,
      dimension,
      shadow.events,
    )))
  }

  /// The counts so far.
  pub fn summary(&self) -> Summary {
    let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
    Summary {
      observed: read(&self.counters.observed),
      created: read(&self.counters.created),
      judged: read(&self.counters.judged),
      agreed: read(&self.counters.agreed),
      only_shadow_rejected: read(&self.counters.only_shadow_rejected),
      only_live_rejected: read(&self.counters.only_live_rejected),
      failed: read(&self.counters.failed),
    }
  }

  /// What the candidate makes of a revision: the engine's own two steps, under the candidate's schema and policy.
  ///
  /// A threshold the request asked for replaces the candidate's as it replaced the live one. Were the candidate
  /// live, the request would have been judged at that threshold too.
  fn judge(&self, observation: &Observation<'_>, judged: &Judged<'_>) -> similarity_core::Result<Evaluation> {
    let policy = match observation.threshold {
      Some(threshold) => self.policy.with_threshold(threshold)?,
      None => self.policy,
    };
    let baseline = judged.baseline;
    let mut evaluation = score::evaluate(
      &self.schema,
      &policy,
      self.dimension,
      &baseline.vectors,
      Some(&baseline.anchor),
      judged.candidate,
    )?;
    evaluation.judge_quantities(
      &self.schema,
      &policy,
      &baseline.quantities,
      &baseline.anchor_quantities,
      judged.quantities,
    );
    Ok(evaluation)
  }

  /// Writes what this shadow is, beside the live schema and policy it runs next to.
  pub fn announce(&self, live_schema: &Schema, live_policy: &Policy) {
    let shadow = self.name.as_str();
    tracing::info!(
      target: RESULT,
      shadow,
      kind = KIND,
      events = events_name(self.events),
      dimension = self.dimension,
      threshold = six(self.policy.threshold),
      anchor_threshold = self.policy.anchor_threshold.map(six),
      reject_on_measure_change = self.policy.reject_on_measure_change,
      live_threshold = six(live_policy.threshold),
      live_anchor_threshold = live_policy.anchor_threshold.map(six),
      live_reject_on_measure_change = live_policy.reject_on_measure_change,
      "shadow_started"
    );
    for field in self.schema.fields() {
      let live = live_schema.field(&field.name);
      tracing::info!(
        target: RESULT,
        shadow,
        field = field.name.as_str(),
        weight = six(field.weight),
        reject_below = field.reject_below.map(six),
        live_weight = live.map(|live| six(live.weight)),
        live_reject_below = live.and_then(|live| live.reject_below).map(six),
        "shadow_schema_field"
      );
    }
    for quantity in self.schema.quantities() {
      let live = live_schema.quantity(&quantity.name);
      tracing::info!(
        target: RESULT,
        shadow,
        quantity = quantity.name.as_str(),
        reject_above_factor = quantity.reject_above_factor,
        reject_incomparable = quantity.reject_incomparable,
        live_reject_above_factor = live.and_then(|live| live.reject_above_factor),
        live_reject_incomparable = live.map(|live| live.reject_incomparable),
        "shadow_schema_quantity"
      );
    }
  }

  /// Writes the counts so far.
  pub fn emit_summary(&self) {
    let summary = self.summary();
    tracing::info!(
      target: RESULT,
      shadow = self.name.as_str(),
      kind = KIND,
      observed = summary.observed,
      created = summary.created,
      judged = summary.judged,
      agreed = summary.agreed,
      only_shadow_rejected = summary.only_shadow_rejected,
      only_live_rejected = summary.only_live_rejected,
      failed = summary.failed,
      "shadow_summary"
    );
  }

  fn emit(&self, sequence: u64, observation: &Observation<'_>, live: &Evaluation, shadow: &Evaluation) {
    let key = format!("{}-{sequence}", self.started);
    let name = self.name.as_str();
    // The thresholds each side was judged at, which are the request's own where it asked for one.
    let threshold = |configured: f32| six(observation.threshold.unwrap_or(configured));
    tracing::info!(
      target: RESULT,
      key = key.as_str(),
      id = observation.id,
      tenant = observation.tenant,
      shadow = name,
      kind = KIND,
      mode = mode_name(observation.mode),
      live_verdict = verdict(live.accepted()),
      shadow_verdict = verdict(shadow.accepted()),
      agree = live.accepted() == shadow.accepted(),
      live_similarity = six(live.comparison.similarity),
      shadow_similarity = six(shadow.comparison.similarity),
      live_anchor_similarity = live.anchor_similarity.map(six),
      shadow_anchor_similarity = shadow.anchor_similarity.map(six),
      live_threshold = six(observation.policy.threshold),
      shadow_threshold = threshold(self.policy.threshold),
      measure_changed = shadow.measure_changed,
      live_stored = observation.stored,
      revision = observation.revision,
      "shadow"
    );
    for (side, reasons) in [("live", &live.reasons), ("shadow", &shadow.reasons)] {
      for reason in reasons {
        emit_reason(&key, name, side, reason);
      }
    }
  }
}

/// Writes `shadow`'s counts every `every`, for as long as it is polled. The server stops it by dropping it, and
/// writes the last counts itself.
pub async fn summarize_every(shadow: Arc<PolicyShadow>, every: Duration) {
  let mut ticks = tokio::time::interval(every);
  // A server that was stalled owes nobody the summaries it missed.
  ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
  // The first tick is immediate, and there is nothing to report yet.
  ticks.tick().await;
  loop {
    ticks.tick().await;
    shadow.emit_summary();
  }
}

fn emit_reason(key: &str, shadow: &str, side: &str, reason: &Reason) {
  tracing::info!(
    target: RESULT,
    key,
    shadow,
    side,
    code = reason.code(),
    subject = reason.subject(),
    value = reason.value().map(six_wide),
    limit = reason.limit().map(six_wide),
    "shadow_reason"
  );
}

impl Observer for PolicyShadow {
  fn observe(&self, observation: &Observation<'_>) {
    bump(&self.counters.observed);
    let Some(judged) = &observation.judged else {
      bump(&self.counters.created);
      return;
    };
    let shadow = match self.judge(observation, judged) {
      Ok(shadow) => shadow,
      Err(error) => {
        bump(&self.counters.failed);
        tracing::warn!(
          shadow = self.name.as_str(),
          id = observation.id,
          %error,
          "the shadow policy could not judge a revision; the live verdict is unaffected"
        );
        return;
      }
    };
    let sequence = bump(&self.counters.judged);
    let live = judged.evaluation;
    let counter = match (live.accepted(), shadow.accepted()) {
      (true, false) => &self.counters.only_shadow_rejected,
      (false, true) => &self.counters.only_live_rejected,
      (true, true) | (false, false) => &self.counters.agreed,
    };
    bump(counter);
    if live.accepted() != shadow.accepted() || self.events == ShadowEvents::All {
      self.emit(sequence, observation, live, &shadow);
    }
  }
}

#[cfg(test)]
mod tests {
  use serde_json::Value;
  use similarity_core::{FieldSpec, Quantity, QuantitySpec};
  use similarity_store::{MemoryStore, StoredRecord, VectorStore};

  use super::*;
  use crate::engine::{Engine, EngineError, EvaluateInput, EvaluateOutput, RecordInput, Verdict, store_layout};
  use crate::worker::TextEmbedder;
  use crate::worker::testing::WordBagEmbedder;

  type TestEngine = Engine<MemoryStore, WordBagEmbedder>;

  const ID: &str = "p1";
  const TENANT: &str = "brand-1";
  const NAME: &str = "blueberry gummies vegan fruit chews";
  /// One word of five is different.
  const RENAMED: &str = "blueberry gummies vegan fruit cartridge";
  const DESCRIPTION: &str = "ten vegan gummies with real fruit";
  const REDESCRIBED: &str = "one gram of pressed rosin";

  /// A schema of a name and a description, by their weights, with a floor for the name and a factor limit for
  /// the one quantity.
  fn schema(weights: (f32, f32), name_floor: Option<f32>, weight_limit: Option<f64>) -> Schema {
    let field = |name: &str, weight: f32, reject_below: Option<f32>| FieldSpec {
      name: name.to_owned(),
      weight,
      required: false,
      reject_below,
    };
    Schema::new(vec![
      field("name", weights.0, name_floor),
      field("description", weights.1, None),
    ])
    .unwrap()
    .with_quantities(vec![QuantitySpec {
      name: "weight".to_owned(),
      reject_above_factor: weight_limit,
      reject_incomparable: false,
    }])
    .unwrap()
  }

  fn plain() -> Schema {
    schema((4.0, 1.0), None, None)
  }

  fn policy(threshold: f32, anchor_threshold: Option<f32>) -> Policy {
    Policy::new(threshold, anchor_threshold).unwrap()
  }

  fn engine(schema: Schema, policy: Policy) -> TestEngine {
    let embedder = WordBagEmbedder::new("word-bag-v1");
    let (layout, dimension) = store_layout(&schema, &embedder);
    Engine::new(MemoryStore::new(layout, dimension), embedder, schema, policy)
  }

  fn request(name: &str, description: &str, grams: Option<f64>) -> EvaluateInput {
    EvaluateInput {
      record: RecordInput {
        id: Some(ID.to_owned()),
        tenant: Some(TENANT.to_owned()),
        fields: vec![
          ("name".to_owned(), name.to_owned()),
          ("description".to_owned(), description.to_owned()),
        ],
        quantities: grams
          .map(|grams| ("weight".to_owned(), Quantity::single(grams, "g")))
          .into_iter()
          .collect(),
      },
      ..EvaluateInput::default()
    }
  }

  /// What a sequence of requests did, with or without a shadow watching.
  struct Ran {
    outputs: Vec<Result<EvaluateOutput, EngineError>>,
    stored: Option<StoredRecord>,
    summary: Summary,
    /// The result lines, parsed.
    results: Vec<Value>,
    logs: Vec<String>,
  }

  impl Ran {
    fn events(&self, name: &str) -> Vec<&Value> {
      self.results.iter().filter(|event| event["message"] == name).collect()
    }

    fn output(&self, index: usize) -> &EvaluateOutput {
      self.outputs[index].as_ref().unwrap()
    }
  }

  /// Runs `requests` through an engine under `live`, watched by `shadow` if there is one. `dimension` is the
  /// width the shadow is told its vectors have, as an offset from the true one.
  fn run_with(
    live: (Schema, Policy),
    shadow: Option<(Schema, Policy, ShadowEvents)>,
    wrong_by: usize,
    requests: &[EvaluateInput],
  ) -> Ran {
    let dimension = WordBagEmbedder::new("word-bag-v1").dimension();
    let shadow = shadow.map(|(schema, policy, events)| {
      Arc::new(PolicyShadow::new(
        "candidate",
        schema,
        policy,
        dimension + wrong_by,
        events,
      ))
    });
    let mut engine = engine(live.0, live.1);
    if let Some(shadow) = &shadow {
      engine = engine.with_observer(shadow.clone());
    }
    let mut outputs = Vec::new();
    let mut stored = None;
    let captured = similarity_log::testing::capture("warn", || {
      let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
      runtime.block_on(async {
        for request in requests {
          outputs.push(engine.evaluate(request.clone()).await);
        }
        stored = engine.store().get(ID).await.unwrap();
      });
    });
    Ran {
      outputs,
      stored,
      summary: shadow.map(|shadow| shadow.summary()).unwrap_or_default(),
      results: captured
        .results
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect(),
      logs: captured.logs,
    }
  }

  fn run(live: (Schema, Policy), shadow: (Schema, Policy, ShadowEvents), requests: &[EvaluateInput]) -> Ran {
    run_with(live, Some(shadow), 0, requests)
  }

  fn alone(live: (Schema, Policy), requests: &[EvaluateInput]) -> Ran {
    run_with(live, None, 0, requests)
  }

  /// Everything about a stored record but the second it was written in.
  fn timeless(record: Option<&StoredRecord>) -> Option<StoredRecord> {
    record.cloned().map(|record| StoredRecord {
      updated_at: 0,
      ..record
    })
  }

  fn number(event: &Value, key: &str) -> f64 {
    event[key]
      .as_f64()
      .unwrap_or_else(|| panic!("`{key}` is not a number in {event}"))
  }

  /// The heart of it: a candidate far stricter than the live policy watches, would have refused an edit, says
  /// so, and the edit is accepted and stored exactly as it is with nobody watching.
  #[test]
  fn a_stricter_candidate_says_what_it_would_have_refused_and_refuses_nothing() {
    let requests = [
      request(NAME, DESCRIPTION, None),
      request(RENAMED, DESCRIPTION, None),
      request(RENAMED, DESCRIPTION, None),
    ];
    let live = || (plain(), policy(0.5, None));
    let watched = run(
      live(),
      (plain(), policy(0.99, None), ShadowEvents::Disagreements),
      &requests,
    );
    let unwatched = alone(live(), &requests);

    assert_eq!(watched.outputs, unwatched.outputs);
    assert_eq!(timeless(watched.stored.as_ref()), timeless(unwatched.stored.as_ref()));
    let edit = watched.output(1);
    assert_eq!((edit.verdict, edit.stored, edit.revision), (Verdict::Accepted, true, 2));
    assert_eq!(watched.stored.as_ref().unwrap().revision, 2);

    assert_eq!(
      watched.summary,
      Summary {
        observed: 3,
        created: 1,
        judged: 2,
        agreed: 1,
        only_shadow_rejected: 1,
        only_live_rejected: 0,
        failed: 0,
      }
    );

    // One line for the disagreement, none for the new record or for the resubmission both accept.
    let events = watched.events("shadow");
    assert_eq!(events.len(), 1, "{:?}", watched.results);
    let event = events[0];
    assert_eq!(event["id"], ID);
    assert_eq!(event["tenant"], TENANT);
    assert_eq!(event["shadow"], "candidate");
    assert_eq!(event["kind"], "policy");
    assert_eq!(event["mode"], "enforce");
    assert_eq!(event["live_verdict"], "accepted");
    assert_eq!(event["shadow_verdict"], "rejected");
    assert_eq!(event["agree"], false);
    assert_eq!(event["live_stored"], true);
    assert_eq!(event["revision"], 2);
    assert_eq!(event["measure_changed"], false);
    let similarity = f64::from(edit.similarity.unwrap());
    assert!((number(event, "live_similarity") - similarity).abs() < 1e-6);
    assert!(
      (number(event, "shadow_similarity") - similarity).abs() < 1e-6,
      "the same weights give the same score"
    );
    assert!((number(event, "live_threshold") - 0.5).abs() < 1e-6);
    assert!((number(event, "shadow_threshold") - 0.99).abs() < 1e-6);
    // The first judged request of this shadow.
    assert!(event["key"].as_str().unwrap().ends_with("-1"), "{event}");

    let reasons = watched.events("shadow_reason");
    assert_eq!(reasons.len(), 1);
    assert_eq!(reasons[0]["key"], event["key"]);
    assert_eq!(reasons[0]["side"], "shadow");
    assert_eq!(reasons[0]["code"], "below_threshold");
    assert!((number(reasons[0], "value") - similarity).abs() < 1e-6);
    assert!((number(reasons[0], "limit") - 0.99).abs() < 1e-6);
    assert!(reasons[0].get("subject").is_none(), "a threshold is about no one field");
  }

  /// The other direction: the live policy refuses, the candidate would not have. Nothing is stored, as with
  /// nobody watching.
  #[test]
  fn a_looser_candidate_says_what_it_would_have_let_through_and_lets_nothing_through() {
    let requests = [request(NAME, DESCRIPTION, None), request(RENAMED, REDESCRIBED, None)];
    let live = || (plain(), policy(0.95, None));
    let watched = run(
      live(),
      (plain(), policy(0.1, None), ShadowEvents::Disagreements),
      &requests,
    );
    let unwatched = alone(live(), &requests);

    assert_eq!(watched.outputs, unwatched.outputs);
    assert_eq!(timeless(watched.stored.as_ref()), timeless(unwatched.stored.as_ref()));
    assert_eq!(
      (watched.output(1).verdict, watched.output(1).stored),
      (Verdict::Rejected, false)
    );
    assert_eq!(
      (
        watched.summary.judged,
        watched.summary.agreed,
        watched.summary.only_live_rejected,
        watched.summary.only_shadow_rejected
      ),
      (1, 0, 1, 0)
    );
    let event = watched.events("shadow")[0];
    assert_eq!(
      (&event["live_verdict"], &event["shadow_verdict"], &event["live_stored"]),
      (&Value::from("rejected"), &Value::from("accepted"), &Value::from(false))
    );
    assert_eq!(event["revision"], 1, "the baseline is where it was");
    let reasons = watched.events("shadow_reason");
    assert_eq!(reasons.len(), 1);
    assert_eq!(
      (&reasons[0]["side"], &reasons[0]["code"]),
      (&Value::from("live"), &Value::from("below_threshold"))
    );
  }

  /// The tenant on a line is the one the request named. A request that named none gets a line without the key.
  #[test]
  fn a_line_names_the_tenant_only_when_the_request_did() {
    let untenanted = |name: &str| {
      let mut request = request(name, DESCRIPTION, None);
      request.record.tenant = None;
      request
    };
    let watched = run(
      (plain(), policy(0.5, None)),
      (plain(), policy(0.99, None), ShadowEvents::Disagreements),
      &[untenanted(NAME), untenanted(RENAMED)],
    );
    let event = watched.events("shadow")[0];
    assert_eq!(event["id"], ID);
    assert!(event.get("tenant").is_none(), "{event}");
  }

  /// A candidate with other weights scores the same vectors differently, against the baseline and against the
  /// anchor alike: exactly as an engine configured that way scores them.
  #[test]
  fn a_candidate_scores_under_its_own_weights_as_an_engine_configured_that_way_would() {
    let requests = [
      request(NAME, DESCRIPTION, None),
      request(RENAMED, DESCRIPTION, None),
      request(RENAMED, REDESCRIBED, None),
    ];
    let lenient = || policy(0.0, Some(0.0));
    let candidate = || schema((1.0, 4.0), None, None);
    let watched = run(
      (plain(), lenient()),
      (candidate(), lenient(), ShadowEvents::All),
      &requests,
    );
    // Both accept everything, so the two engines hold the same history.
    let reference = alone((candidate(), lenient()), &requests);

    let events = watched.events("shadow");
    assert_eq!(events.len(), 2, "every judged request, agreed or not");
    for (event, index) in events.iter().zip([1, 2]) {
      let (live, shadow) = (watched.output(index), reference.output(index));
      assert_eq!(event["agree"], true);
      assert!((number(event, "live_similarity") - f64::from(live.similarity.unwrap())).abs() < 1e-6);
      assert!((number(event, "shadow_similarity") - f64::from(shadow.similarity.unwrap())).abs() < 1e-6);
      assert!((number(event, "live_anchor_similarity") - f64::from(live.anchor_similarity.unwrap())).abs() < 1e-6);
      assert!((number(event, "shadow_anchor_similarity") - f64::from(shadow.anchor_similarity.unwrap())).abs() < 1e-6);
    }
    // The weights matter here: the name moved a little and the description a lot.
    let last = events[1];
    assert!(number(last, "shadow_anchor_similarity") < number(last, "live_anchor_similarity") - 0.1);
    assert!(watched.events("shadow_reason").is_empty());
    assert_eq!((watched.summary.judged, watched.summary.agreed), (2, 2));
  }

  /// Floors, drift, and quantity limits are the candidate's own too, and each gives its own code.
  #[test]
  fn a_candidate_applies_its_own_floor_drift_threshold_and_quantity_limit() {
    let requests = [
      request(NAME, DESCRIPTION, Some(1.0)),
      request(RENAMED, DESCRIPTION, Some(10.0)),
    ];
    let strict = schema((4.0, 1.0), Some(0.95), Some(2.0));
    let watched = run(
      (plain(), policy(0.5, None)),
      (strict, policy(0.5, Some(0.99)), ShadowEvents::Disagreements),
      &requests,
    );
    assert_eq!(watched.output(1).verdict, Verdict::Accepted);
    let mut reasons: Vec<(&str, Option<&str>)> = watched
      .events("shadow_reason")
      .iter()
      .map(|reason| {
        assert_eq!(reason["side"], "shadow");
        (reason["code"].as_str().unwrap(), reason["subject"].as_str())
      })
      .collect();
    reasons.sort_unstable();
    assert_eq!(
      reasons,
      [
        ("anchor_drift", None),
        ("field_below_floor", Some("name")),
        ("quantity_beyond_factor", Some("weight")),
      ]
    );
    let factor = watched
      .events("shadow_reason")
      .into_iter()
      .find(|reason| reason["code"] == "quantity_beyond_factor")
      .unwrap();
    assert_eq!((number(factor, "value"), number(factor, "limit")), (10.0, 2.0));
  }

  /// A request that names its own threshold is judged at it by both. That is what the candidate would have done
  /// with the request, were it live.
  #[test]
  fn a_request_s_own_threshold_replaces_the_candidate_s_as_it_replaces_the_live_one() {
    let asked = |threshold: f32| EvaluateInput {
      threshold: Some(threshold),
      ..request(RENAMED, DESCRIPTION, None)
    };
    let live = || (plain(), policy(0.5, None));
    let candidate = || (plain(), policy(0.99, None), ShadowEvents::All);

    let low = run(live(), candidate(), &[request(NAME, DESCRIPTION, None), asked(0.2)]);
    assert_eq!((low.summary.agreed, low.summary.only_shadow_rejected), (1, 0));
    let event = low.events("shadow")[0];
    assert_eq!(event["shadow_verdict"], "accepted");
    assert!((number(event, "live_threshold") - 0.2).abs() < 1e-6);
    assert!((number(event, "shadow_threshold") - 0.2).abs() < 1e-6);

    let high = run(live(), candidate(), &[request(NAME, DESCRIPTION, None), asked(0.999)]);
    assert_eq!(high.output(1).verdict, Verdict::Rejected);
    assert_eq!((high.summary.agreed, high.summary.only_live_rejected), (1, 0));
    assert_eq!(high.events("shadow_reason").len(), 2, "one reason on each side");
  }

  #[test]
  fn a_dry_run_and_a_forced_write_are_judged_and_named_as_what_they_were() {
    let moded = |mode: WriteMode| EvaluateInput {
      mode,
      ..request(RENAMED, REDESCRIBED, None)
    };
    let requests = [
      request(NAME, DESCRIPTION, None),
      moded(WriteMode::DryRun),
      moded(WriteMode::Force),
    ];
    let watched = run(
      (plain(), policy(0.95, None)),
      (plain(), policy(0.1, None), ShadowEvents::Disagreements),
      &requests,
    );
    let events = watched.events("shadow");
    let seen: Vec<_> = events
      .iter()
      .map(|event| {
        (
          event["mode"].as_str().unwrap(),
          event["live_stored"].as_bool().unwrap(),
          event["revision"].as_u64().unwrap(),
        )
      })
      .collect();
    assert_eq!(seen, [("dry_run", false, 1), ("force", true, 2)]);
    assert!(events[0]["key"].as_str().unwrap().ends_with("-1"));
    assert!(events[1]["key"].as_str().unwrap().ends_with("-2"));
    assert_ne!(events[0]["key"], events[1]["key"]);
  }

  /// A candidate that cannot judge (here it has been told the wrong vector width, which the configuration cannot
  /// produce) fails by itself: counted, logged, and the request is answered and stored as ever.
  #[test]
  fn a_candidate_that_cannot_judge_is_counted_and_the_request_is_answered_as_ever() {
    let requests = [request(NAME, DESCRIPTION, None), request(RENAMED, DESCRIPTION, None)];
    let live = || (plain(), policy(0.5, None));
    let broken = run_with(
      live(),
      Some((plain(), policy(0.99, None), ShadowEvents::All)),
      1,
      &requests,
    );
    let unwatched = alone(live(), &requests);
    assert_eq!(broken.outputs, unwatched.outputs);
    assert_eq!(timeless(broken.stored.as_ref()), timeless(unwatched.stored.as_ref()));
    assert_eq!(
      broken.summary,
      Summary {
        observed: 2,
        created: 1,
        failed: 1,
        ..Summary::default()
      }
    );
    assert!(broken.events("shadow").is_empty());
    assert_eq!(broken.logs.len(), 1, "{:?}", broken.logs);
    assert!(broken.logs[0].contains("could not judge"), "{}", broken.logs[0]);

    // A threshold the request asks for is validated before anything is judged, so an invalid one never reaches
    // the candidate.
    let invalid = EvaluateInput {
      threshold: Some(7.0),
      ..request(RENAMED, DESCRIPTION, None)
    };
    let refused = run(
      live(),
      (plain(), policy(0.99, None), ShadowEvents::All),
      &[request(NAME, DESCRIPTION, None), invalid],
    );
    assert!(refused.outputs[1].is_err());
    assert_eq!((refused.summary.observed, refused.summary.failed), (1, 0));
  }

  /// Everything a shadow writes over a short life: its announcement, a new record, a disagreement, its counts.
  fn everything_written() -> (Vec<String>, Vec<Value>) {
    let requests = [
      request(NAME, DESCRIPTION, Some(1.0)),
      request(RENAMED, REDESCRIBED, Some(10.0)),
    ];
    let live_schema = schema((4.0, 1.0), Some(0.3), Some(20.0));
    let live_policy = policy(0.2, Some(0.1));
    let candidate = PolicyShadow::new(
      "candidate",
      schema((2.5, 1.0), None, Some(2.0)),
      policy(0.9, None),
      WordBagEmbedder::new("word-bag-v1").dimension(),
      ShadowEvents::All,
    );
    let candidate = Arc::new(candidate);
    let engine = engine(live_schema.clone(), live_policy).with_observer(candidate.clone());
    let captured = similarity_log::testing::capture("off", || {
      candidate.announce(&live_schema, &live_policy);
      let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
      runtime.block_on(async {
        for request in &requests {
          engine.evaluate(request.clone()).await.unwrap();
        }
      });
      candidate.emit_summary();
    });
    let events = captured
      .results
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();
    (captured.results, events)
  }

  /// D15 and D17: one flat object per line, scalars only, and never a word of the record.
  #[test]
  fn what_a_shadow_writes_is_flat_and_carries_no_record_text() {
    let (lines, events) = everything_written();
    for (line, event) in lines.iter().zip(&events) {
      for (key, value) in event.as_object().unwrap() {
        assert!(!value.is_object() && !value.is_array(), "`{key}` is nested in {line}");
        assert!(
          !value.is_null(),
          "`{key}` is null in {line}: an absent value is left out"
        );
      }
      for word in [NAME, RENAMED, DESCRIPTION, REDESCRIBED]
        .iter()
        .flat_map(|text| text.split(' '))
        .filter(|word| word.len() > 3)
      {
        assert!(!line.contains(word), "`{word}` of a record's text is in {line}");
      }
    }

    let messages: Vec<&str> = events.iter().map(|event| event["message"].as_str().unwrap()).collect();
    assert_eq!(
      messages,
      [
        "shadow_started",
        "shadow_schema_field",
        "shadow_schema_field",
        "shadow_schema_quantity",
        "shadow",
        "shadow_reason",
        "shadow_reason",
        "shadow_summary",
      ]
    );
  }

  /// A file of shadow events says what produced it: the candidate's settings beside the live ones, and the
  /// counts at the end.
  #[test]
  fn a_shadow_announces_its_settings_beside_the_live_ones_and_reports_its_counts() {
    let (_, events) = everything_written();
    let named = |name: &str| -> Vec<&Value> { events.iter().filter(|event| event["message"] == name).collect() };

    let started = named("shadow_started")[0];
    assert_eq!(
      (&started["shadow"], &started["kind"], &started["events"]),
      (&Value::from("candidate"), &Value::from("policy"), &Value::from("all"))
    );
    assert!((number(started, "threshold") - 0.9).abs() < 1e-6);
    assert!(
      started.get("anchor_threshold").is_none(),
      "the candidate enforces no drift"
    );
    assert!((number(started, "live_threshold") - 0.2).abs() < 1e-6);
    assert!((number(started, "live_anchor_threshold") - 0.1).abs() < 1e-6);
    assert_eq!(
      (
        &started["reject_on_measure_change"],
        &started["live_reject_on_measure_change"]
      ),
      (&Value::from(false), &Value::from(false))
    );

    let name = named("shadow_schema_field")
      .into_iter()
      .find(|field| field["field"] == "name")
      .unwrap();
    assert_eq!((number(name, "weight"), number(name, "live_weight")), (2.5, 4.0));
    assert!(name.get("reject_below").is_none());
    assert!((number(name, "live_reject_below") - 0.3).abs() < 1e-6);
    let weight = named("shadow_schema_quantity")[0];
    assert_eq!(weight["quantity"], "weight");
    assert_eq!(
      (
        number(weight, "reject_above_factor"),
        number(weight, "live_reject_above_factor")
      ),
      (2.0, 20.0)
    );
    assert_eq!(
      (&weight["reject_incomparable"], &weight["live_reject_incomparable"]),
      (&Value::from(false), &Value::from(false))
    );

    let summary = named("shadow_summary")[0];
    let count = |key: &str| summary[key].as_u64().unwrap();
    assert_eq!(
      (
        count("observed"),
        count("created"),
        count("judged"),
        count("agreed"),
        count("only_shadow_rejected"),
        count("only_live_rejected"),
        count("failed")
      ),
      (2, 1, 1, 0, 1, 0, 0)
    );
    assert_eq!(
      (&summary["shadow"], &summary["kind"]),
      (&Value::from("candidate"), &Value::from("policy"))
    );
  }

  const SHADOWED: &str = r#"
    [policy]
    threshold = 0.5

    [fields.name]
    weight = 4.0
    reject_below = 0.3

    [fields.description]
    weight = 1.0

    [quantities.weight]

    [shadow]
    name = "stricter-name"
    events = "all"
    summary_every_s = 1

    [shadow.policy]
    threshold = 0.9

    [shadow.fields.name]
    reject_below = "off"

    [shadow.quantities.weight]
    reject_above_factor = 2.0
  "#;

  /// The whole path from a configuration file: the candidate it describes, announced beside the live settings.
  #[test]
  fn a_shadow_is_made_from_the_configuration_and_none_without_the_table() {
    let dimension = WordBagEmbedder::new("word-bag-v1").dimension();
    let config = Config::parse(SHADOWED).unwrap();
    let shadow = PolicyShadow::from_config(&config, dimension).unwrap().unwrap();
    assert_eq!(
      (shadow.name.as_str(), shadow.events, shadow.dimension),
      ("stricter-name", ShadowEvents::All, dimension)
    );
    assert_eq!(shadow.policy.threshold, 0.9);
    assert_eq!(shadow.schema.field("name").unwrap().reject_below, None);
    assert_eq!(shadow.schema.field("name").unwrap().weight, 4.0);
    assert_eq!(shadow.schema.quantity("weight").unwrap().reject_above_factor, Some(2.0));
    assert_eq!(shadow.summary(), Summary::default());

    let without = SHADOWED.split("[shadow]").next().unwrap();
    assert!(
      PolicyShadow::from_config(&Config::parse(without).unwrap(), dimension)
        .unwrap()
        .is_none()
    );
  }

  /// The counts are written on a timer, not at once, and they are the counts at the time.
  #[test]
  fn the_counts_are_written_again_and_again_while_the_server_runs() {
    let shadow = Arc::new(PolicyShadow::new(
      "candidate",
      plain(),
      policy(0.9, None),
      1,
      ShadowEvents::Disagreements,
    ));
    let captured = similarity_log::testing::capture("off", || {
      let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
      runtime.block_on(async {
        let ticking = tokio::spawn(summarize_every(shadow.clone(), Duration::from_millis(20)));
        tokio::task::yield_now().await;
        bump(&shadow.counters.observed);
        tokio::time::sleep(Duration::from_millis(150)).await;
        ticking.abort();
      });
    });
    let summaries: Vec<Value> = captured
      .results
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();
    // At least two in seven intervals. No upper bound: on a busy machine the sleep runs long, and the timer
    // keeps its pace through it.
    assert!(summaries.len() >= 2, "{summaries:?}");
    for summary in &summaries {
      assert_eq!(summary["message"], "shadow_summary");
      assert_eq!(summary["observed"], 1);
    }
  }
}
