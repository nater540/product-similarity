//! Recorded runs, read back: for comparing two of them and for a report of several.
//!
//! `similarity-runs` writes a run's directory and reads its manifest (D23). This module reads what is in the
//! directory: a training run's curve and measurements ([`training`]), or a calibration
//! ([`crate::offline::events`]). It lives in this crate because the calibration reader and the calibration
//! comparison do, and the trainer must not link them.
//!
//! A resumed training run is several directories, each naming the one it continues. Naming any of them loads the
//! chain up to it, so that a run is one curve however many times it was interrupted.

use std::path::Path;

use anyhow::{Context, bail};
use similarity_runs::{EVENTS_FILE, KIND_CALIBRATE, KIND_TRAIN, Listed};

use crate::offline::events;

pub mod compare;
pub mod html;
pub mod training;

/// How many directories a resume chain may have before it is taken for a loop.
const CHAIN_LIMIT: usize = 64;

/// A run directory, loaded.
#[derive(Debug)]
pub enum Loaded {
  /// A training run: its directories, oldest first, ending with the one that was named, and what they hold.
  Training {
    /// The directories of the resume chain.
    chain: Vec<Listed>,
    /// The run, stitched from them.
    run: Box<training::TrainingRun>,
  },
  /// A calibration.
  Calibration {
    /// The directory.
    listed: Box<Listed>,
    /// The calibration it holds.
    run: Box<events::Run>,
  },
}

impl Loaded {
  /// The run that was named: for a training run, the last of its chain.
  pub fn listed(&self) -> Option<&Listed> {
    match self {
      Self::Training { chain, .. } => chain.last(),
      Self::Calibration { listed, .. } => Some(listed.as_ref()),
    }
  }

  /// The run's id.
  pub fn id(&self) -> &str {
    self.listed().map_or("", |listed| listed.manifest.id.as_str())
  }
}

fn events_of(listed: &Listed) -> anyhow::Result<String> {
  let path = listed.dir.join(EVENTS_FILE);
  std::fs::read_to_string(&path).with_context(|| format!("cannot read {}", path.display()))
}

/// Loads the run in `dir`: a calibration, or a training run with every run it continues.
///
/// # Errors
///
/// Returns an error if the directory is not a run, if it is of a kind this cannot read, if its events cannot be
/// read, or if a run it continues is no longer beside it.
pub fn load(dir: &Path) -> anyhow::Result<Loaded> {
  let named = similarity_runs::open(dir)?;
  match named.manifest.kind.as_str() {
    KIND_CALIBRATE => {
      let run = events::read(&events_of(&named)?).with_context(|| format!("in {}", dir.display()))?;
      Ok(Loaded::Calibration {
        listed: Box::new(named),
        run: Box::new(run),
      })
    }
    KIND_TRAIN => {
      // Newest first while following the links back, then turned round.
      let mut chain = vec![named];
      while let Some(continues) = chain.last().and_then(|listed| listed.manifest.continues.clone()) {
        // A checkpoint written without a recorded run is where the record of the run begins.
        let Some(earlier) = continues.run else { break };
        if chain.len() >= CHAIN_LIMIT {
          bail!(
            "{} continues more than {CHAIN_LIMIT} runs; the links go in a circle",
            dir.display()
          );
        }
        let beside = chain
          .last()
          .and_then(|listed| listed.dir.parent())
          .map_or_else(|| Path::new(&earlier).to_owned(), |parent| parent.join(&earlier));
        let listed = similarity_runs::open(&beside)
          .with_context(|| format!("{} continues the run `{earlier}`, which cannot be read", dir.display()))?;
        chain.push(listed);
      }
      chain.reverse();
      let mut parts = Vec::with_capacity(chain.len());
      for listed in &chain {
        let part = training::read(&events_of(listed)?).with_context(|| format!("in {}", listed.dir.display()))?;
        parts.push((part, listed.manifest.continues.as_ref().map(|continues| continues.step)));
      }
      // The oldest directory's own `continues`, if it has one, points outside the record: it starts the curve.
      if let Some(first) = parts.first_mut() {
        first.1 = None;
      }
      Ok(Loaded::Training {
        run: Box::new(training::stitch(parts)),
        chain,
      })
    }
    other => bail!("{} is a `{other}` run, which this version cannot read", dir.display()),
  }
}
