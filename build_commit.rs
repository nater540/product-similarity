// Shared by the build scripts of the two binary crates, which `include!` it. It tells the binary which source
// commit it was built from, so that a model directory can record what wrote it (`provenance.json`).
//
// The answer comes from the `RECORD_SIMILARITY_COMMIT` environment variable when that is set (the Docker build
// has no `.git` to ask and passes it in), otherwise from `git`, and otherwise it is `unknown`. Nothing here can
// fail a build: a checkout without git, or a source tarball, still compiles.

/// Runs `git` in the crate's directory and returns what it printed, if it ran and succeeded.
fn git(arguments: &[&str]) -> Option<String> {
  let output = std::process::Command::new("git").args(arguments).output().ok()?;
  output
    .status
    .success()
    .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// Sets `RECORD_SIMILARITY_BUILD_COMMIT` and `RECORD_SIMILARITY_BUILD_DIRTY` (`true`, `false`, or empty when it
/// is not known) for the crate being built.
fn emit_commit() {
  println!("cargo:rerun-if-env-changed=RECORD_SIMILARITY_COMMIT");
  // Whether the tree is dirty can change without any file of this crate changing, and so can the commit. Naming
  // a path that never exists makes cargo run this script on every build; it costs two short `git` calls, and
  // the crate is only rebuilt when an answer actually changes.
  println!("cargo:rerun-if-changed=.the-commit-is-checked-on-every-build");

  let given = std::env::var("RECORD_SIMILARITY_COMMIT")
    .ok()
    .map(|commit| commit.trim().to_owned())
    .filter(|commit| !commit.is_empty());
  let (commit, dirty) = match given {
    Some(commit) => (commit, String::new()),
    None => match git(&["rev-parse", "--short=12", "HEAD"]).filter(|commit| !commit.is_empty()) {
      Some(commit) => {
        let dirty = git(&["status", "--porcelain"]).map_or_else(String::new, |status| (!status.is_empty()).to_string());
        (commit, dirty)
      }
      None => ("unknown".to_owned(), String::new()),
    },
  };
  println!("cargo:rustc-env=RECORD_SIMILARITY_BUILD_COMMIT={commit}");
  println!("cargo:rustc-env=RECORD_SIMILARITY_BUILD_DIRTY={dirty}");
}
