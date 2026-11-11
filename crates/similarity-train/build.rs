//! Tells the training binary which source commit it was built from. See `build_commit.rs` at the workspace root.

include!("../../build_commit.rs");

fn main() {
  emit_commit();
}
