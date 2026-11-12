//! A match calibration for a person, and the same calibration as data.
//!
//! Both are drawn from one [`MatchSummary`](super::stats::matching::MatchSummary), so the report and the events
//! cannot disagree. The events are flat, like a calibration's (D17): one `match_calibration` line, the schema,
//! then one `match_menu_item` line per menu item with a `match_field` and `match_quantity` line per item, joined on
//! `menu_item_id`, and a `match_summary` at the end. Their names are an interface once written: fields are added,
//! never renamed. No menu item's text is in any of them.

use std::fmt::Write as _;

use similarity_core::{FieldChange, Schema};
use similarity_log::RESULT;

use super::drift_report::Model;
use super::events::{emit_schema, six, six_wide};
use super::matching::MatchCalibration;
use super::quantity_change_name;
use super::stats::matching::MatchSummary;
use super::stats::{Range, Rate};

fn percent(rate: &Rate) -> String {
  let Some(value) = rate.rate else {
    return "-".to_owned();
  };
  let interval = rate.interval.map_or_else(String::new, |interval| {
    format!(" ({:.1}% to {:.1}%)", interval.low * 100.0, interval.high * 100.0)
  });
  format!("{:.1}%{interval}", value * 100.0)
}

fn counted(rate: &Rate) -> String {
  format!("{} of {}  {}", rate.errors, rate.of, percent(rate))
}

fn range(range: Option<Range>) -> String {
  range.map_or_else(
    || "-".to_owned(),
    |range| format!("{:.3} to {:.3} ({})", range.low, range.high, range.count),
  )
}

/// Which side has a field: both, the menu item alone, or the product alone. A match reads a comparison with the
/// product as the baseline, so `Added` is the menu item's and `Removed` the product's.
fn presence(change: FieldChange) -> &'static str {
  match change {
    FieldChange::Added | FieldChange::Restored => "menu_item",
    FieldChange::Removed => "product",
    FieldChange::Unchanged | FieldChange::Modified => "both",
  }
}

fn render_menu_items(calibration: &MatchCalibration, out: &mut String) {
  // Writing to a `String` cannot fail, here and below.
  let _ = writeln!(
    out,
    "{:>4} {:>6} {:>10} {:>7}  {:<32} best other",
    "rank", "own", "best other", "margin", "group"
  );
  for scored in &calibration.menu_items {
    let rank = scored
      .own_rank
      .map_or_else(|| format!(">{}", calibration.depth), |rank| rank.to_string());
    let own = scored.own.comparison.similarity;
    let (other, margin, name) = scored.best_other.as_ref().map_or_else(
      || ("-".to_owned(), "-".to_owned(), String::new()),
      |other| {
        let similarity = other.ranked.comparison.similarity;
        let sibling = if other.same_brand { " (same brand)" } else { "" };
        (
          format!("{similarity:.3}"),
          format!("{:+.3}", own - similarity),
          format!("{}{sibling}", other.ranked.id),
        )
      },
    );
    let lost = if scored.overfetch_lost { "  << over-fetch" } else { "" };
    let _ = writeln!(
      out,
      "{rank:>4} {own:>6.3} {other:>10} {margin:>7}  {:<32} {name}{lost}",
      scored.menu_item.group
    );
  }
}

fn render_ranking(calibration: &MatchCalibration, summary: &MatchSummary, out: &mut String) {
  let _ = writeln!(out, "\nranking");
  let _ = writeln!(out, "  own product first          {}", counted(&summary.recall_at_1));
  let _ = writeln!(out, "  own product in the first 5 {}", counted(&summary.recall_at_5));
  let _ = writeln!(
    out,
    "  own product in the first {:<2} {}",
    calibration.depth,
    counted(&summary.recall_at_depth)
  );
  if let Some(mrr) = summary.mean_reciprocal_rank {
    let _ = writeln!(out, "  mean reciprocal rank       {mrr:.3}");
  }
  let _ = writeln!(
    out,
    "  first, scoring every product {}",
    counted(&summary.exhaustive_recall_at_1)
  );
  let _ = writeln!(
    out,
    "  served ranking differs from that {}  (what an over-fetch of {} loses)",
    counted(&summary.overfetch_lost),
    calibration.overfetch
  );
  if let Some(brands) = summary.brands {
    let _ = writeln!(out, "  intervals are the middle 95% over resamples of {brands} brands");
  }
}

fn render_scores(summary: &MatchSummary, out: &mut String) {
  let _ = writeln!(out, "\nscores");
  let _ = writeln!(out, "  own product        {}", range(summary.own));
  let _ = writeln!(out, "  best other product {}", range(summary.best_other));
  let _ = writeln!(
    out,
    "  best other is a product of the same brand {}",
    counted(&summary.siblings)
  );
  if let Some(auc) = summary.auc {
    let _ = writeln!(out, "  own above best other, over all pairings (AUC) {auc:.3}");
  }
}

fn render_threshold(summary: &MatchSummary, out: &mut String) {
  let _ = writeln!(out, "\nthreshold");
  match summary.suggestion {
    Some(suggestion) => {
      let _ = writeln!(
        out,
        "  best separates the two at {:.3} (margin {:.3}): {} own below it, {} best other at or above it",
        suggestion.threshold, suggestion.margin, suggestion.false_rejects, suggestion.false_accepts
      );
    }
    None => {
      let _ = writeln!(out, "  no suggestion: it takes menu items and at least two products");
    }
  }
  for point in &summary.operating_points {
    let (Some(threshold), Some(accept), Some(reject)) = (point.threshold, &point.false_accept, &point.false_reject)
    else {
      let _ = writeln!(
        out,
        "  no threshold keeps wrong matches of an absent product to {:.0}%",
        point.target * 100.0
      );
      continue;
    };
    let _ = writeln!(
      out,
      "  to match a menu item whose product is absent at most {:.0}% of the time: {threshold:.3}, where {} would be, \
       and {} of menu items score their own product below it",
      point.target * 100.0,
      percent(accept),
      percent(reject)
    );
  }
  match &summary.at_threshold {
    Some(at) => {
      let _ = writeln!(out, "  configured: {:.3}", at.threshold);
      let _ = writeln!(out, "    own product below it        {}", counted(&at.own_below));
      let _ = writeln!(out, "    best other at or above it   {}", counted(&at.other_reaches));
      let _ = writeln!(out, "    matched to the wrong product {}", counted(&at.wrong));
    }
    None => {
      let _ = writeln!(
        out,
        "  configured: none (`matching.threshold`), so `Match` ranks and decides nothing"
      );
    }
  }
}

fn render_quantities_and_left_out(calibration: &MatchCalibration, summary: &MatchSummary, out: &mut String) {
  let input = &calibration.input;
  if !summary.quantities.is_empty() {
    let _ = writeln!(
      out,
      "\n{:<18} {:>8} {:>6} {:>12} {:>12} {:>12}",
      "quantity", "compared", "same", "incomparable", "menu item only", "product only"
    );
    for quantity in &summary.quantities {
      let _ = writeln!(
        out,
        "{:<18} {:>8} {:>6} {:>12} {:>12} {:>12}",
        quantity.quantity,
        quantity.compared,
        quantity.same,
        quantity.incomparable,
        quantity.only_menu_item,
        quantity.only_product
      );
    }
  }

  let mut left_out = Vec::new();
  if input.skipped_split > 0 {
    left_out.push(format!("{} menu items of other splits", input.skipped_split));
  }
  if input.duplicate_menu_items > 0 {
    left_out.push(format!(
      "{} menu items identical to another of their group",
      input.duplicate_menu_items
    ));
  }
  for ((key, reason), count) in &input.ignored {
    left_out.push(format!("{count} values of `{key}` ({reason})"));
  }
  if !left_out.is_empty() {
    let _ = writeln!(out, "\nleft out: {}", left_out.join("; "));
  }
}

/// Renders a match calibration as a report, for a terminal. Every number is from `summary`.
pub fn render_match(calibration: &MatchCalibration, summary: &MatchSummary) -> String {
  let mut out = String::new();
  let input = &calibration.input;
  let split = input
    .split
    .as_ref()
    .map_or_else(String::new, |split| format!(" of the `{split}` split"));
  let _ = writeln!(
    out,
    "menu items  {}{split}, scored against {} products ({} from the catalog export)",
    summary.menu_items, calibration.products, input.distractors
  );
  let _ = writeln!(
    out,
    "served    the first {} candidates, from the {} the search returns ({} for each)\n",
    calibration.depth,
    calibration.depth.saturating_mul(calibration.overfetch),
    calibration.overfetch
  );
  render_menu_items(calibration, &mut out);

  render_ranking(calibration, summary, &mut out);
  render_scores(summary, &mut out);
  render_threshold(summary, &mut out);
  render_quantities_and_left_out(calibration, summary, &mut out);
  out
}

fn parts(rate: &Rate) -> (Option<f64>, Option<f64>, Option<f64>) {
  (
    rate.rate.map(six_wide),
    rate.interval.map(|interval| six_wide(interval.low)),
    rate.interval.map(|interval| six_wide(interval.high)),
  )
}

fn emit_menu_items(calibration: &MatchCalibration, threshold: Option<f32>) {
  for scored in &calibration.menu_items {
    let menu_item = &scored.menu_item;
    let own = scored.own.comparison.similarity;
    let other = scored.best_other.as_ref();
    let best_other = other.map(|other| other.ranked.comparison.similarity);
    // What `Match` would say at the configured threshold: the first of the served ranking, if it reaches it.
    let top_is_own = scored.own_rank == Some(1);
    let top = if top_is_own { Some(own) } else { best_other };
    let matched = threshold.map(|threshold| top.is_some_and(|top| top >= threshold));
    tracing::info!(
      target: RESULT,
      menu_item_id = menu_item.id.as_str(),
      group = menu_item.group.as_str(),
      brand_id = menu_item.brand_id.as_deref(),
      split = menu_item.split.as_deref(),
      own_rank = scored.own_rank,
      exhaustive_own_rank = scored.exhaustive_own_rank,
      own_similarity = six(own),
      own_search_similarity = six(scored.own.search_similarity),
      best_other = other.map(|other| other.ranked.id.as_str()),
      best_other_similarity = best_other.map(six),
      best_other_same_brand = other.map(|other| other.same_brand),
      margin = best_other.map(|best_other| six(own) - six(best_other)).map(six_wide),
      overfetch_lost = scored.overfetch_lost,
      matched,
      correct = matched.map(|matched| matched && top_is_own),
      "match_menu_item"
    );
    for field in &scored.own.comparison.fields {
      let against_other = other.and_then(|other| {
        other
          .ranked
          .comparison
          .fields
          .iter()
          .find(|theirs| theirs.field == field.field)
          .and_then(|theirs| theirs.similarity)
      });
      tracing::info!(
        target: RESULT,
        menu_item_id = menu_item.id.as_str(),
        field = field.field.as_str(),
        presence = presence(field.change),
        weight = six(field.weight),
        own_similarity = field.similarity.map(six),
        best_other_similarity = against_other.map(six),
        "match_field"
      );
    }
    for quantity in &scored.own.quantities {
      tracing::info!(
        target: RESULT,
        menu_item_id = menu_item.id.as_str(),
        quantity = quantity.field.as_str(),
        change = quantity_change_name(quantity.change),
        factor = quantity.factor,
        incomparable = quantity.incomparable,
        zero_crossing = quantity.zero_crossing,
        "match_quantity"
      );
    }
  }
}

/// Writes a match calibration as events: what it was, the schema, every menu item, and the summary.
pub fn emit_match(
  model: Model<'_>,
  schema: &Schema,
  calibration: &MatchCalibration,
  summary: &MatchSummary,
  threshold: Option<f32>,
  seed: u64,
) {
  let input = &calibration.input;
  tracing::info!(
    target: RESULT,
    fingerprint = model.fingerprint,
    dimension = model.dimension,
    max_tokens = model.max_tokens,
    threshold = threshold.map(six),
    depth = calibration.depth,
    overfetch = calibration.overfetch,
    groups_sha256 = input.sha256.as_str(),
    split = input.split.as_deref(),
    products = calibration.products,
    distractors = input.distractors,
    menu_items = calibration.menu_items.len(),
    skipped_split = input.skipped_split,
    duplicate_menu_items = input.duplicate_menu_items,
    seed,
    "match_calibration"
  );
  emit_schema(schema);
  emit_menu_items(calibration, threshold);
  for quantity in &summary.quantities {
    tracing::info!(
      target: RESULT,
      quantity = quantity.quantity.as_str(),
      compared = quantity.compared,
      same = quantity.same,
      incomparable = quantity.incomparable,
      only_menu_item = quantity.only_menu_item,
      only_product = quantity.only_product,
      "match_quantity_summary"
    );
  }
  for point in &summary.operating_points {
    let (accept, accept_low, accept_high) = point.false_accept.as_ref().map_or((None, None, None), parts);
    let (reject, reject_low, reject_high) = point.false_reject.as_ref().map_or((None, None, None), parts);
    tracing::info!(
      target: RESULT,
      target_rate = point.target,
      threshold = point.threshold.map(six),
      other_reaches_rate = accept,
      other_reaches_low = accept_low,
      other_reaches_high = accept_high,
      own_below_rate = reject,
      own_below_low = reject_low,
      own_below_high = reject_high,
      "match_operating_point"
    );
  }
  for ((key, reason), count) in &input.ignored {
    tracing::info!(target: RESULT, key = key.as_str(), reason = %reason, count, "ignored_key");
  }

  let (recall_at_1, recall_at_1_low, recall_at_1_high) = parts(&summary.recall_at_1);
  let (recall_at_5, recall_at_5_low, recall_at_5_high) = parts(&summary.recall_at_5);
  let (recall_at_depth, recall_at_depth_low, recall_at_depth_high) = parts(&summary.recall_at_depth);
  let at = summary.at_threshold.as_ref();
  tracing::info!(
    target: RESULT,
    menu_items = summary.menu_items,
    brands = summary.brands,
    recall_at_1,
    recall_at_1_low,
    recall_at_1_high,
    recall_at_5,
    recall_at_5_low,
    recall_at_5_high,
    recall_at_depth,
    recall_at_depth_low,
    recall_at_depth_high,
    exhaustive_recall_at_1 = summary.exhaustive_recall_at_1.rate.map(six_wide),
    mean_reciprocal_rank = summary.mean_reciprocal_rank.map(six_wide),
    overfetch_lost = summary.overfetch_lost.errors,
    own_low = summary.own.map(|range| six_wide(range.low)),
    own_high = summary.own.map(|range| six_wide(range.high)),
    best_other_low = summary.best_other.map(|range| six_wide(range.low)),
    best_other_high = summary.best_other.map(|range| six_wide(range.high)),
    best_other_same_brand = summary.siblings.errors,
    auc = summary.auc.map(six_wide),
    suggested_threshold = summary.suggestion.map(|suggestion| six(suggestion.threshold)),
    suggested_margin = summary.suggestion.map(|suggestion| six(suggestion.margin)),
    suggested_own_below = summary.suggestion.map(|suggestion| suggestion.false_rejects),
    suggested_other_reaches = summary.suggestion.map(|suggestion| suggestion.false_accepts),
    own_below = at.map(|at| at.own_below.errors),
    other_reaches = at.map(|at| at.other_reaches.errors),
    wrong = at.map(|at| at.wrong.errors),
    "match_summary"
  );
}

#[cfg(test)]
mod tests {
  use serde_json::Value;
  use similarity_core::testing::tiny_embedder;
  use similarity_core::{EmbedderOptions, FieldSpec, Policy, QuantitySpec};

  use super::super::Offline;
  use super::super::matching::{calibrate_match, parse_groups};
  use super::super::stats::matching::summarize_match;
  use super::*;

  const GROUPS: &str = r#"
    {"group": "gummies", "brand_id": "brand-1", "split": "holdout", "catalog": {"name": "blueberry gummies", "category": "edibles", "weight": {"value": 40, "unit": "g"}}, "menu_items": [{"name": "wyld blueberry gummies mg", "strain": "indica"}, {"name": "blueberry gummies", "weight": {"value": 40000, "unit": "mg"}}]}
    {"group": "vape", "brand_id": "brand-1", "split": "holdout", "catalog": {"name": "sativa vape cartridge", "category": "vape"}, "menu_items": [{"name": "vape cartridge sativa resin"}]}
    {"group": "flower", "brand_id": "brand-2", "split": "calibrate", "catalog": {"name": "hybrid flower"}, "menu_items": [{"name": "flower hybrid indica", "category": "flower"}]}
  "#;

  fn offline() -> Offline {
    let field = |name: &str, weight: f32| FieldSpec {
      name: name.to_owned(),
      weight,
      required: false,
      reject_below: None,
    };
    let schema = Schema::new(vec![field("name", 4.0), field("category", 2.0)])
      .unwrap()
      .with_quantities(vec![QuantitySpec {
        name: "weight".to_owned(),
        reject_above_factor: None,
        reject_incomparable: false,
      }])
      .unwrap();
    let options = EmbedderOptions {
      max_tokens: 16,
      batch_size: 8,
      output_dim: None,
    };
    Offline::new(tiny_embedder(&options), schema, Policy::new(0.5, None).unwrap())
  }

  /// A calibration of the holdout menu items, summarized at a threshold, with its report and its events.
  fn written(threshold: Option<f32>) -> (MatchCalibration, MatchSummary, String, Vec<String>, Vec<Value>) {
    let offline = offline();
    let groups = parse_groups(GROUPS, None, offline.schema(), Some("holdout")).unwrap();
    let calibration = calibrate_match(&offline, groups, 2, 5).unwrap();
    let summary = summarize_match(&calibration, threshold, 7);
    let report = render_match(&calibration, &summary);
    let model = Model {
      fingerprint: offline.fingerprint(),
      dimension: offline.dimension(),
      max_tokens: offline.max_tokens(),
    };
    let captured = similarity_log::testing::capture("off", || {
      emit_match(model, offline.schema(), &calibration, &summary, threshold, 7);
    });
    let events = captured
      .results
      .iter()
      .map(|line| serde_json::from_str(line).unwrap())
      .collect();
    (calibration, summary, report, captured.results, events)
  }

  fn named<'a>(events: &'a [Value], name: &str) -> Vec<&'a Value> {
    events.iter().filter(|event| event["message"] == name).collect()
  }

  /// D15 and D17: one flat object per line, an absent value left out, and no word of a menu item or a product.
  #[test]
  fn match_events_are_flat_joined_on_the_menu_item_and_carry_no_text() {
    let (calibration, _, _, lines, events) = written(Some(0.5));
    for (line, event) in lines.iter().zip(&events) {
      for (key, value) in event.as_object().unwrap() {
        assert!(
          !value.is_object() && !value.is_array() && !value.is_null(),
          "`{key}` in {line}"
        );
      }
      for word in ["blueberry", "cartridge", "wyld", "hybrid", "sativa", "resin"] {
        assert!(!line.contains(word), "`{word}` is in {line}");
      }
    }
    let messages: Vec<&str> = events.iter().map(|event| event["message"].as_str().unwrap()).collect();
    assert_eq!(
      messages[..4],
      ["match_calibration", "schema_field", "schema_field", "schema_quantity"]
    );
    assert_eq!(messages.last(), Some(&"match_summary"));

    let header = named(&events, "match_calibration")[0];
    assert_eq!(
      (
        header["split"].as_str(),
        header["products"].as_u64(),
        header["menu_items"].as_u64(),
        header["depth"].as_u64(),
        header["overfetch"].as_u64(),
        header["skipped_split"].as_u64(),
        header["seed"].as_u64(),
      ),
      (Some("holdout"), Some(3), Some(3), Some(2), Some(5), Some(1), Some(7))
    );
    assert!((header["threshold"].as_f64().unwrap() - 0.5).abs() < 1e-6);
    assert_eq!(header["groups_sha256"].as_str().unwrap().len(), 64);

    let menu_items = named(&events, "match_menu_item");
    assert_eq!(menu_items.len(), 3);
    for (event, scored) in menu_items.iter().zip(&calibration.menu_items) {
      assert_eq!(event["menu_item_id"], scored.menu_item.id.as_str());
      assert_eq!(event["group"], scored.menu_item.group.as_str());
      assert_eq!(event["brand_id"], "brand-1");
      assert_eq!(event["split"], "holdout");
      assert_eq!(
        event["own_rank"].as_u64(),
        scored.own_rank.map(|rank| u64::try_from(rank).unwrap())
      );
      assert_eq!(
        event["exhaustive_own_rank"].as_u64(),
        Some(scored.exhaustive_own_rank as u64)
      );
      let own = f64::from(scored.own.comparison.similarity);
      assert!((event["own_similarity"].as_f64().unwrap() - own).abs() < 1e-6);
      let other = scored.best_other.as_ref().unwrap();
      assert_eq!(event["best_other"], other.ranked.id.as_str());
      assert_eq!(event["best_other_same_brand"], other.same_brand);
      let best_other = f64::from(other.ranked.comparison.similarity);
      assert!((event["margin"].as_f64().unwrap() - (own - best_other)).abs() < 1e-5);
      // Matched means the first served candidate reaches the threshold; correct, that it is the menu item's own.
      let top = if scored.own_rank == Some(1) { own } else { best_other };
      assert_eq!(event["matched"], top >= 0.5);
      assert_eq!(event["correct"], top >= 0.5 && scored.own_rank == Some(1));
    }
    // Every field and quantity line belongs to a menu item.
    let ids: Vec<&str> = menu_items
      .iter()
      .map(|event| event["menu_item_id"].as_str().unwrap())
      .collect();
    let fields = named(&events, "match_field");
    assert!(!fields.is_empty());
    for line in fields.iter().chain(&named(&events, "match_quantity")) {
      assert!(ids.contains(&line["menu_item_id"].as_str().unwrap()));
    }
    // The first menu item has a name and its product a name and a category.
    let first: Vec<(&str, &str, bool)> = fields
      .iter()
      .filter(|field| field["menu_item_id"] == ids[0])
      .map(|field| {
        (
          field["field"].as_str().unwrap(),
          field["presence"].as_str().unwrap(),
          field.get("own_similarity").is_some(),
        )
      })
      .collect();
    assert_eq!(first, [("category", "product", false), ("name", "both", true)]);
    let quantity = named(&events, "match_quantity")
      .into_iter()
      .find(|line| line["menu_item_id"] == ids[1])
      .unwrap();
    assert_eq!(
      (quantity["quantity"].as_str(), quantity["change"].as_str()),
      (Some("weight"), Some("unchanged"))
    );
    let ignored = named(&events, "ignored_key");
    assert_eq!((ignored.len(), ignored[0]["key"].as_str()), (1, Some("strain")));
  }

  /// The report and the events are two renderings of one summary.
  #[test]
  fn the_match_report_and_the_events_give_the_same_numbers() {
    let (calibration, summary, report, _, events) = written(Some(0.5));
    let line = named(&events, "match_summary")[0];
    let count = |key: &str| line[key].as_u64().map(|count| usize::try_from(count).unwrap());
    assert_eq!(count("menu_items"), Some(summary.menu_items));
    assert_eq!(count("overfetch_lost"), Some(summary.overfetch_lost.errors));
    assert_eq!(count("best_other_same_brand"), Some(summary.siblings.errors));
    let at = summary.at_threshold.unwrap();
    assert_eq!(
      (count("own_below"), count("other_reaches"), count("wrong")),
      (
        Some(at.own_below.errors),
        Some(at.other_reaches.errors),
        Some(at.wrong.errors)
      )
    );
    let close = |key: &str, value: Option<f64>| match (line[key].as_f64(), value) {
      (Some(found), Some(wanted)) => assert!((found - wanted).abs() < 1e-6, "{key}: {found} and {wanted}"),
      (found, wanted) => assert_eq!(found, wanted, "{key}"),
    };
    close("recall_at_1", summary.recall_at_1.rate);
    close("recall_at_5", summary.recall_at_5.rate);
    close("recall_at_depth", summary.recall_at_depth.rate);
    close("mean_reciprocal_rank", summary.mean_reciprocal_rank);
    close("auc", summary.auc);
    close("own_low", summary.own.map(|range| range.low));
    close("best_other_high", summary.best_other.map(|range| range.high));
    close(
      "suggested_threshold",
      summary.suggestion.map(|suggestion| f64::from(suggestion.threshold)),
    );
    assert_eq!(
      line.get("brands"),
      None,
      "one brand: no intervals, and no count of brands"
    );
    assert_eq!(line.get("recall_at_1_low"), None);

    // And the same numbers are in the text.
    assert!(
      report
        .starts_with("menu items  3 of the `holdout` split, scored against 3 products (0 from the catalog export)\n")
    );
    assert!(report.contains("served    the first 2 candidates, from the 10 the search returns (5 for each)"));
    let counted = |rate: &Rate| format!("{} of {}", rate.errors, rate.of);
    assert!(
      report.contains(&format!("own product first          {}", counted(&summary.recall_at_1))),
      "{report}"
    );
    assert!(
      report.contains(&format!("own product below it        {}", counted(&at.own_below))),
      "{report}"
    );
    assert!(
      report.contains(&format!("matched to the wrong product {}", counted(&at.wrong))),
      "{report}"
    );
    assert!(report.contains("configured: 0.500"), "{report}");
    assert!(
      report.contains("left out: 1 menu items of other splits; 1 values of `strain` (not in the schema)"),
      "{report}"
    );
    assert!(report.contains("weight                    1      1"), "{report}");
    for scored in &calibration.menu_items {
      assert!(report.contains(&format!("{:.3}", scored.own.comparison.similarity)));
    }
    assert!(
      !report.contains("blueberry"),
      "a report names groups, never a menu item's text"
    );

    // Without a configured threshold nothing is said to be matched, in either rendering.
    let (_, _, report, _, events) = written(None);
    assert!(report.contains("configured: none"), "{report}");
    let line = named(&events, "match_summary")[0];
    assert_eq!((line.get("own_below"), line.get("wrong")), (None, None));
    assert!(
      named(&events, "match_menu_item")
        .iter()
        .all(|event| event.get("matched").is_none())
    );
    assert_eq!(named(&events, "match_calibration")[0].get("threshold"), None);
  }
}
